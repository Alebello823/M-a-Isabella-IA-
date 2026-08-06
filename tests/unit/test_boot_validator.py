from bootstrap.boot_validator import BootValidator

def test_validation_passes():
    v = BootValidator()
    result = v.validate()
    assert result.success
    assert len(result.errors) == 0

def test_diagnostics_have_python():
    v = BootValidator()
    result = v.validate()
    assert "python_version" in result.diagnostics
    assert "sqlite" in result.diagnostics
