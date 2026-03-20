use anyhow::Result;
use assert_cmd::Command;

#[test]
fn test_terrapulse_cli_help() -> Result<()> {
    let mut cmd = Command::cargo_bin("terrapulse")?;
    let assert = cmd.arg("--help").assert();
    assert.success()
          .stdout(predicates::str::contains("Fast TerraPulse inference pipeline"));
    Ok(())
}

#[test]
fn test_pipeline_dry_run() -> Result<()> {
    Ok(())
}
