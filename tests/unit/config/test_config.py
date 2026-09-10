from config.config import CONFIG


def test_config_loads() -> None:
    """Test that application configuration loads successfully."""

    assert CONFIG

    assert "aws" in CONFIG

    aws_config = CONFIG["aws"]

    assert "region" in aws_config

    aws_region = str(aws_config["region"])

    assert aws_region

    print("AWS_REGION:", aws_region)
