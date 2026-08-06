import logging
import time
from pathlib import Path
from typing import Optional

from bootstrap.configuration_loader import ConfigurationLoader
from bootstrap.dependency_container import DependencyContainer
from bootstrap.filesystem_initializer import FileSystemInitializer
from bootstrap.boot_validator import BootValidator
from bootstrap.signal_manager import SignalManager
from config.models import Settings
from hardware import detect_hardware, CapabilityMapper
from internal.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)

class Kernel:
    def __init__(self, boot_id: str = "unknown"):
        self.boot_id = boot_id
        self.settings: Optional[Settings] = None
        self.container: Optional[DependencyContainer] = None
        self._running = False
        self.hardware_profile = None

    def boot(self) -> bool:
        # Cargar configuración
        config_loader = ConfigurationLoader()
        self.settings = config_loader.load()

        # Validar entorno
        validator = BootValidator()
        result = validator.validate()
        if not result.success:
            for err in result.errors:
                logger.error("Validation error: %s", err)
            return False
        for warn in result.warnings:
            logger.warning("Validation warning: %s", warn)

        # Inicializar sistema de archivos
        fs_init = FileSystemInitializer(self.settings)
        if not fs_init.initialize():
            logger.error("No se pudo inicializar el sistema de archivos")
            return False

        # Detectar hardware y seleccionar perfil
        if self.settings.hardware.auto_detect:
            hw_info = detect_hardware()
            logger.info("Hardware detectado: RAM disp=%s GB, CPU=%s", hw_info.available_ram_gb, hw_info.cpu_name)
            mapper = CapabilityMapper(Path(PROJECT_ROOT / "config" / "hardware_profiles"))
            self.hardware_profile = mapper.select_profile(
                hw_info,
                manual_tier=self.settings.hardware.manual_profile
            )
        else:
            # Si no se auto-detecta, usar el perfil manual o uno por defecto
            self.hardware_profile = {
                "tier": "manual",
                "recommended_model": str(self.settings.inference.default_model_path),
                "max_tokens": 512,
                "temperature": 0.7,
                "enable_vector_memory": False,
                "enable_knowledge_extraction": False,
            }
            if self.settings.hardware.manual_profile:
                mapper = CapabilityMapper(Path(PROJECT_ROOT / "config" / "hardware_profiles"))
                self.hardware_profile = mapper.select_profile(
                    detect_hardware(), manual_tier=self.settings.hardware.manual_profile
                )

        # Contenedor de dependencias
        self.container = DependencyContainer()
        self.container.register_instance(Settings, self.settings)

        # Registrar servicios principales
        self._register_core_services()

        # Señales
        SignalManager(self.shutdown)

        self._running = True
        logger.info("Kernel arrancado correctamente")
        return True

    def _register_core_services(self):
        # Motor de inferencia (con parámetros del perfil)
        from inference.engine import LlamaCppEngine
        model_path = self.hardware_profile.get("recommended_model", self.settings.inference.default_model_path)
        engine = LlamaCppEngine(
            model_path=model_path,
            max_tokens=self.hardware_profile.get("max_tokens", 512),
            temperature=self.hardware_profile.get("temperature", 0.7),
        )
        self.container.register_instance("InferenceEngineProtocol", engine)

        # Repositorio episódico base
        from memory.storage.episodic_repository import EpisodicRepository
        episodic_repo = EpisodicRepository()
        self.container.register_instance("EpisodicRepository", episodic_repo)

        # Repositorio vectorial (condicional)
        vector_repo = None
        if self.hardware_profile.get("enable_vector_memory", False):
            from memory.storage.vector_repository import VectorRepository
            vector_repo = VectorRepository()
            self.container.register_instance("VectorRepository", vector_repo)
            # Inyectar vector_repo en el repositorio episódico para búsqueda híbrida
            episodic_repo.vector_repo = vector_repo
            logger.info("Memoria vectorial activada")

        # Repositorio semántico (hechos)
        semantic_repo = None
        if self.hardware_profile.get("enable_knowledge_extraction", False):
            from memory.storage.semantic_repository import SemanticRepository
            semantic_repo = SemanticRepository()
            self.container.register_instance("SemanticRepository", semantic_repo)
            logger.info("Extracción de conocimiento activada")

        # Extractor de conocimiento (si hay semántico)
        knowledge_extractor = None
        if semantic_repo:
            from memory.curation.knowledge_extractor import KnowledgeExtractor
            knowledge_extractor = KnowledgeExtractor(
                llm_engine=engine,
                semantic_repo=semantic_repo
            )
            self.container.register_instance("KnowledgeExtractor", knowledge_extractor)

        # Orquestador (con todas las dependencias opcionales)
        from core.orchestrator import Orchestrator
        orchestrator = Orchestrator(
            inference_engine=engine,
            episodic_repo=episodic_repo,
            knowledge_extractor=knowledge_extractor,
            vector_repo=vector_repo,
        )
        self.container.register_instance("Orchestrator", orchestrator)

    def run(self, mode: str = "sync"):
        while self._running:
            time.sleep(1)

    def shutdown(self):
        logger.info("Apagando kernel...")
        self._running = False
