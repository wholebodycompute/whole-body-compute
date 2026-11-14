class WholeBodyCompute < Formula
  desc "Local AST context shield, vision & memory runtime for AI coding agents"
  homepage "https://wholebodycompute.org"
  url "https://files.pythonhosted.org/packages/source/w/whole-body-compute/whole_body_compute-0.1.0.tar.gz"
  sha256 "90dc11d816b7d0d1665a0231770380cbc43f6ee15cf68f32fbf42a581aa019e2"
  license :cannot_represent
  head "https://github.com/wholebodycompute/whole-body-compute.git", branch: "main"

  depends_on "python@3.12"
  depends_on "uv"

  def install
    venv_dir = libexec/"venv"
    python_bin = Formula["python@3.12"].opt_bin/"python3.12"

    system "uv", "venv", venv_dir, "--python", python_bin
    if ENV["HOMEBREW_WBC_WHEEL_DIR"] && !ENV["HOMEBREW_WBC_WHEEL_DIR"].empty?
      system "uv", "pip", "install", "--python", venv_dir/"bin/python",
             "--no-index", "--find-links", ENV["HOMEBREW_WBC_WHEEL_DIR"], buildpath
    else
      system "uv", "pip", "install", "--python", venv_dir/"bin/python", buildpath
    end

    %w[body wbc body-immune-hook foveal-vision-mcp].each do |exe|
      bin.install_symlink venv_dir/"bin"/exe if (venv_dir/"bin"/exe).exist?
    end
  end

  def caveats
    <<~EOS
      Awaken Whole Body Compute in any coding workspace in <10 seconds:
        body begin

      Optional macOS background LaunchAgents (Foveal Socket & :3030 Memory HTTP):
        body daemon install

      Upgrade to Whole Body Compute Pro ($29/mo) or Team ($79/seat/mo):
        body license --activate <POLAR_KEY>
        https://wholebodycompute.com/pricing
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/body --version")
    assert_match version.to_s, shell_output("#{bin}/wbc --version")

    hook_out = pipe_output("#{bin}/body-immune-hook", '{"toolCall":{"name":"run_command","args":{"CommandLine":"echo ok"}}}')
    assert_match "allow", hook_out

    (testpath/"workspace").mkpath
    ENV["VOICEFI_LICENSE_PATH"] = (testpath/"license.json").to_s
    ENV["WBC_DATA_DIR"] = (testpath/"wbc_data").to_s
    ENV["VOICEFI_IMMUNE_AUDIT_PATH"] = (testpath/"immune_audit.jsonl").to_s
    ENV["VOICEFI_MOCK_VAGUS"] = "1"
    ENV["VOICEFI_MOCK_AX"] = "1"
    ENV["VOICEFI_MOCK_SILICON"] = "1"
    ENV["VOICEFI_MOCK_PROPRIOCEPTION"] = "1"

    begin_json = shell_output("cd #{testpath}/workspace && #{bin}/body begin --skip-audit --json")
    assert_match '"wiring"', begin_json
    assert_match '"hooks_status"', begin_json
  end
end
