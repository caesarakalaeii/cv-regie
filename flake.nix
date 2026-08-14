{
  # Keep this line accurate and one line long: `nix flake metadata` prints it,
  # and it is the first thing a cold agent learns about the repo.
  description = "cv-regie -- OpenCV/YOLO/DeepFace multi-camera director that picks and frames the best feed. Run `nix flake show` for the command map.";

  # nixpkgs is the only input, on purpose.
  #
  # flake-utils would buy exactly one thing here -- eachDefaultSystem -- which is
  # the three-line genAttrs below. In exchange it costs a second lock node, a
  # second upstream that can break, and a hardcoded system list this repo cannot
  # edit. That list is currently broken: it still contains x86_64-darwin, which
  # now throws (see `systems` below).
  #
  # nixos-unstable is the same channel the author's own NixOS config tracks, so
  # `nix develop` here and `nixos-rebuild` there resolve the same store paths and
  # share one cache.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    # `...` rather than a closed { self, nixpkgs }: adding a second input later
    # would otherwise fail with "called with unexpected argument 'self'".
    #
    # `self` is not decoration: it is the only way a wrapper in the store can
    # name this repo's own files, which is what anchors every verb (see
    # rootPreamble). It does mean the wrappers rebuild whenever a tracked file
    # changes -- five shellcheck runs, about a second, and worth it.
    { self, nixpkgs, ... }:
    let
      lib = nixpkgs.lib;

      # x86_64-darwin is deliberately absent: nixpkgs 26.11 replaced that whole
      # attribute set with a `throw`. genAttrs is lazy, so plain `nix develop` on
      # Linux would not notice -- it detonates on `nix flake check --all-systems`.
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      forAllSystems = f: lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      # ======================================================================
      # PER-REPO BLOCK 1 -- the toolchain
      # ======================================================================
      # Deliberately small: this repo's real dependency set (torch, tensorflow,
      # ultralytics, deepface, opencv-python) lives in requirements.txt and is
      # installed into .venv by `dev-setup`. Nix owns the interpreter and the
      # launcher; pip owns the wheels. See PER-REPO BLOCK 2 for why that split
      # still needs help from nix on NixOS.
      #
      # python313 because that is what the pinned requirement set resolves for:
      # verified `uv pip compile requirements.txt --python-version 3.13` picks
      # torch 2.13.0 / tensorflow 2.21.0 / opencv-python 5.0.0.93, all as wheels.
      # Do not switch to the rolling `python3` alias -- when it moves, every
      # .venv in the fleet is invalidated on the same afternoon.
      toolchain = pkgs: [
        # ---- this repo's ecosystem ----
        pkgs.python313
        pkgs.uv
        pkgs.ruff

        # ---- present in every repo in the fleet ----
        pkgs.git
        pkgs.jq
        pkgs.gnumake
      ];

      # ======================================================================
      # PER-REPO BLOCK 2 -- libraries that get dlopened, not linked
      # ======================================================================
      # Every entry below was found by running the failing import, not guessed.
      # With an empty LD_LIBRARY_PATH, in a venv built by `dev-setup`:
      #
      #   import numpy -> libstdc++.so.6      -> stdenv.cc.cc.lib
      #   import cv2   -> libxcb.so.1         -> libxcb
      #                -> libGL.so.1          -> libGL
      #                -> libgthread-2.0.so.0 -> glib
      #
      # and then cv.imshow (the whole point of this repo) additionally loads the
      # Qt5 xcb platform plugin out of the wheel, whose own missing deps are
      # libX11.so.6, libXext.so.6, libSM.so.6 and libICE.so.6. Verified end to
      # end: a real cv2 window opened and closed with this list in place.
      #
      # Do NOT "fix" this by adding pkgs.opencv4 or pkgs.ffmpeg: the
      # opencv-python wheel ships its own libopencv_*.so, its own Qt5 5.15.19 and
      # its own FFmpeg (`cv2.getBuildInformation()` -> "GUI: QT5", "FFMPEG: YES")
      # and finds them by RPATH. A nixpkgs OpenCV next to it is several hundred
      # megabytes that nothing loads. If you ever drop opencv-python from
      # requirements.txt in favour of nixpkgs' python313Packages.opencv4, delete
      # this list too -- but note that ultralytics, deepface, tf-keras,
      # pyvirtualcam and lap are not all packaged, so the venv cannot go away.
      #
      # Two things this cannot fix, so do not expect it to:
      #   * pyvirtualcam needs the v4l2loopback KERNEL MODULE on Linux. That is
      #     host config (boot.extraModulePackages), not flake config.
      #   * requirements.txt pulls the CUDA-enabled torch wheels (nvidia-cublas,
      #     nvidia-cudnn-cu13, ...). Measured on a cold cache: 3.1 GiB
      #     downloaded, 7.1 GB of .venv on disk. They run CPU-only out of the
      #     box (verified `torch.cuda.is_available() == False` with no drivers);
      #     actually using the GPU needs the host's driver libs on the loader
      #     path (programs.nix-ld / hardware.nvidia), which no project flake can
      #     supply.
      nativeLibs = pkgs: [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
        pkgs.libxcb
        pkgs.libGL
        pkgs.glib
        pkgs.libX11
        pkgs.libXext
        pkgs.libSM
        pkgs.libICE
      ];

      # ======================================================================
      # PER-REPO BLOCK 3 -- constant environment variables
      # ======================================================================
      # Constants only. Anything that must READ an existing value
      # (LD_LIBRARY_PATH), UNSET something (SOURCE_DATE_EPOCH) or touch the work
      # tree goes in the shellHook. This attrset is applied to BOTH surfaces --
      # the dev shell and every `nix run` wrapper -- so a command cannot behave
      # differently depending on how it was invoked.
      envVars =
        pkgs:
        {
          # Keep uv on the nix interpreter. Left alone it downloads its own
          # portable CPython, which then resolves a different set of wheels than
          # this shell pins: two Pythons, one venv, no way to tell which is live.
          UV_PYTHON = "${pkgs.python313}/bin/python";
          UV_PYTHON_DOWNLOADS = "never";
          # /nix/store and the work tree are usually different filesystems, so
          # uv's default hardlink strategy warns on every single install.
          UV_LINK_MODE = "copy";
          PIP_DISABLE_PIP_VERSION_CHECK = "1";
          # tensorflow (via deepface) logs its CPU-feature and oneDNN probes at
          # INFO on every import, which is pure noise in an agent's context. 1
          # filters INFO only -- warnings and errors still print. Do not raise it
          # to 2 or 3; that starts hiding real problems.
          TF_CPP_MIN_LOG_LEVEL = "1";
        }
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          # The opencv-python wheel bundles exactly ONE Qt platform plugin, xcb
          # (verified: Qt reports "Available platform plugins are: xcb"). On a
          # Wayland session Qt tries "wayland" first, cannot find it, prints
          # `qt.qpa.plugin: Could not find the Qt platform plugin "wayland"` and
          # only then falls back to xcb -- which works through Xwayland, but only
          # because the libs in PER-REPO BLOCK 2 are on the loader path. Naming
          # xcb up front skips both the warning and the fallback. Linux-only: the
          # darwin wheel uses cocoa, where this value would break imshow.
          QT_QPA_PLATFORM = "xcb";
        };

      # ======================================================================
      # PER-REPO BLOCK 4 -- the command map
      # ======================================================================
      # THE single source of truth. It generates `apps` (so `nix run .#run`
      # works), the `dev-*` wrappers on PATH inside the shell, and `dev-help`.
      #
      # No `test` and no `build` verb: this repo has no test suite and no
      # packaging step, and a stub that echoes "not applicable" would turn
      # `nix flake show` into a liar. Absence is information.
      commands = pkgs: {
        setup = {
          # --allow-existing is not cosmetic: without it a second `dev-setup`
          # -- the obvious move after editing requirements.txt -- dies with
          # "A virtual environment already exists at: .venv" and exit 2, before
          # the install line ever runs. Verified against uv 0.12.3. Do not
          # "fix" that with --clear instead: that throws away a 7 GB venv to
          # add one package.
          description = "(network, 3 GB download / 7 GB on disk) create/update .venv from requirements.txt";
          # The .venv belongs to a checkout, and the store snapshot is read-only,
          # so there is nothing sensible to do without one -- least of all
          # unpacking 7 GB into whichever directory the caller happened to be in.
          text = ''
            require_work_tree
            uv venv --allow-existing "$REPO_ROOT/.venv"
            uv pip install --python "$REPO_ROOT/.venv/bin/python" -r "$REPO_ROOT/requirements.txt" "$@"
          '';
        };
        lint = {
          description = "ruff check (the whole repo, from any directory)";
          # `cd` first, then a bare `.` default. Both halves are load-bearing:
          # `ruff check "$@"` alone checked the caller's cwd, and even
          # `ruff check "''${@:-$SOMEROOT}"` still checks the cwd the moment the
          # caller passes a flag rather than a path (`--fix`, `--select F401`),
          # because any argument suppresses the default. Standing in the root
          # closes both, and it makes a relative path argument mean the same thing
          # no matter where the command was invoked from.
          #
          # ruff's incremental cache lands in $PWD. In the snapshot branch that is
          # the read-only store, so it is switched off there -- five files do not
          # need a cache, and littering the caller's directory with .ruff_cache
          # was part of the same bug.
          text = ''
            if [ -n "$REPO_ROOT" ]; then
              cd "$REPO_ROOT"
              ruff check "''${@:-.}"
            else
              cd "$SRC_ROOT"
              ruff check --no-cache "''${@:-.}"
            fi
          '';
        };
        fmt = {
          description = "ruff format (rewrites files, so it needs the checkout)";
          # MUTATING, hence no $SRC_ROOT fallback: formatting the snapshot would
          # either fail on the read-only store or, worse, report "1 file
          # reformatted" for a change nobody can ever see. And no cwd default --
          # that is exactly how `nix run /path/to/this-repo#fmt` used to rewrite
          # Python that had nothing to do with this project.
          text = ''
            require_work_tree
            cd "$REPO_ROOT"
            ruff format "''${@:-.}"
          '';
        };
        run = {
          # The venv interpreter by absolute path, not a bare `python`: the
          # wrappers prepend the nix toolchain to PATH, so a bare name resolves
          # to the store copy and misses everything `setup` installed.
          #
          # main.py resolves "model/pose/yolov8n-pose.pt" and "./database"
          # against the CURRENT directory, not against the script, which is why
          # this cds to the root instead of merely naming main.py absolutely:
          # started from a subdirectory (or from anywhere at all, via the flake
          # URL) it looked for the pose weights beside the caller and created a
          # ./database there. It opens webcam ports [0, 1] and downloads
          # DeepFace/YOLO weights into $HOME on first run.
          description = "(network on first run, needs webcams) start the director";
          text = ''
            require_work_tree
            cd "$REPO_ROOT"
            "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/main.py" "$@"
          '';
        };
      };

      # ======================================================================
      # GENERIC MACHINERY -- byte-identical across the fleet, do not edit
      # ======================================================================

      # Prepend, never assign: a host LD_LIBRARY_PATH may be carrying something
      # the user needs, and clobbering it breaks binaries they launch from here.
      # Linux only -- on darwin the loader variable is DYLD_*, and exporting a
      # Linux-shaped value there is at best useless.
      ldPreamble =
        pkgs:
        lib.optionalString (pkgs.stdenv.hostPlatform.isLinux && nativeLibs pkgs != [ ]) ''
          export LD_LIBRARY_PATH="${lib.makeLibraryPath (nativeLibs pkgs)}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';

      # Every command gets two anchors, and NEITHER of them is the caller's cwd.
      #
      #   $SRC_ROOT   this flake's own source tree as copied into the store when
      #               the wrapper was built: always present, always exactly this
      #               repo's content, always read-only. It is the only repo path
      #               `nix run /elsewhere/this-repo#lint` can be certain of -- the
      #               wrapper is a store path and has no idea where the checkout
      #               it came from lives. It sees git-tracked files only, so a
      #               brand new file is invisible until `git add`.
      #   $REPO_ROOT  the live checkout, or EMPTY when the caller is not standing
      #               in it. Preferred whenever it exists: it is writable and it
      #               sees edits the snapshot does not.
      #
      # The previous `git rev-parse --show-toplevel || pwd` was worse than no
      # anchor at all. From an unrelated directory it resolved to that directory,
      # so `nix run <url>#lint` -- the form CI and a cold agent use -- reported
      # "All checks passed!" having inspected zero of this repo's files, and
      # `nix run <url>#fmt` rewrote a stranger's source. `git rev-parse` on its
      # own is not enough either: run from inside some OTHER checkout it happily
      # reports that repo. So a candidate only counts as ours when every
      # top-level name in the snapshot also exists in it -- cheap, needs no tool
      # beyond the shell, and unlike comparing flake.nix it survives editing this
      # file.
      #
      # Read-only verbs then fall back to $SRC_ROOT and report the same thing from
      # any cwd. Verbs that write or keep state call `require_work_tree` and
      # refuse instead: the snapshot is read-only, and the caller's directory is
      # not ours to guess at.
      rootPreamble = ''
        SRC_ROOT=${self}
        REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
        if [ -n "$REPO_ROOT" ]; then
          for entry in "$SRC_ROOT"/*; do
            [ -e "$REPO_ROOT/''${entry##*/}" ] || { REPO_ROOT=""; break; }
          done
        fi
        export SRC_ROOT REPO_ROOT

        # Called by every verb that writes, before it writes anything.
        require_work_tree() {
          if [ -z "$REPO_ROOT" ]; then
            echo "''${0##*/}: this verb writes to the checkout, and the directory" >&2
            echo "  you called from is not one. Run it from inside the work tree," >&2
            echo "  or from a \`nix develop\` started there." >&2
            exit 1
          fi
        }
      '';

      # One derivation per command, reused by both `apps` and the dev shell, so
      # the two can never diverge. `dev-` prefixed because a bare `test` binary
      # earlier on PATH would shadow the POSIX shell builtin and quietly break
      # every script in the repo that uses it.
      wrappers =
        pkgs:
        lib.mapAttrs (
          name: cmd:
          pkgs.writeShellApplication {
            name = "dev-${name}";
            runtimeInputs = toolchain pkgs;
            runtimeEnv = envVars pkgs;
            meta.description = cmd.description;
            text = ''
              ${rootPreamble}
              ${ldPreamble pkgs}
              ${cmd.text}
            '';
          }
        ) (commands pkgs);

      helpFor =
        pkgs:
        let
          cmds = commands pkgs;
          names = lib.attrNames cmds;
          width = lib.foldl' (a: n: lib.max a (builtins.stringLength n)) 0 names;
          pad = n: n + lib.concatStrings (lib.genList (_: " ") (width - builtins.stringLength n));
          line = n: c: "  dev-${pad n}  ${c.description}";
        in
        pkgs.writeShellApplication {
          name = "dev-help";
          meta.description = "print this repo's command map (works offline)";
          text = ''
            cat <<'EOF'
            ${lib.concatStringsSep "\n" (lib.mapAttrsToList line cmds)}
            EOF
          '';
        };
    in
    {
      # `nix flake show` -- the discovery entrypoint, and deliberately the whole
      # machine-facing contract: every app carries a meta.description, which
      # `nix flake show` prints inline and `nix flake show --json` exposes at
      # .apps.<system>.<name>.description. Pure evaluation, so an agent gets the
      # entire command map in one cheap call without reading a README.
      apps = forAllSystems (
        pkgs:
        lib.mapAttrs (name: cmd: {
          type = "app";
          program = "${(wrappers pkgs).${name}}/bin/dev-${name}";
          meta.description = cmd.description;
        }) (commands pkgs)
      );

      # `nix develop` -- the toolchain, plus a dev-<verb> for every app.
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = toolchain pkgs ++ lib.attrValues (wrappers pkgs) ++ [ (helpFor pkgs) ];

          env = envVars pkgs;

          # Some C extensions compile at -O0, where glibc's _FORTIFY_SOURCE
          # becomes a hard error instead of a warning.
          hardeningDisable = [ "fortify" ];

          shellHook = ''
            # mkShell inherits SOURCE_DATE_EPOCH=315532800 (1980-01-01) from
            # stdenv, and any wheel or zip built in here then dies with "ZIP does
            # not support timestamps before 1980".
            unset SOURCE_DATE_EPOCH

            ${rootPreamble}
            ${ldPreamble pkgs}

            # Nothing networked, stateful or interactive above this line, and
            # nothing below it either. No venv creation, no weight downloads.
            # Bootstrapping here would make a cold `nix develop -c dev-lint`
            # pull 3 GB of wheels before running anything, on EVERY invocation --
            # the exact failure an unattended agent cannot diagnose. That is what
            # `dev-setup` is for, and its description says so.

            # The banner is interactive-only, and this guard is load-bearing:
            # shellHook output lands on the STDOUT of `nix develop -c <cmd>`, so
            # an unguarded echo corrupts anything parsing it. $- is the only
            # reliable discriminator -- it lacks `i` for `nix develop -c` and has
            # it at an interactive prompt. Do not test $PS1 (unset in both) or
            # $IN_NIX_SHELL (set in both). >&2 is the second layer, for a caller
            # that runs us on a pty.
            case $- in
              *i*) echo "cv-regie dev shell -- 'dev-help' for the command map" >&2 ;;
            esac
          '';
        };
      });

      # `nix flake check` -- honest by construction. It realises the toolchain
      # closure (so a typo'd or currently-broken attr fails here) and builds
      # every wrapper, which runs shellcheck over every command text. NEVER add a
      # check that always passes: an agent reads "all checks passed!" as a
      # signal, and a fake check makes `nix flake check` a liar.
      checks = forAllSystems (pkgs: {
        toolchain =
          pkgs.runCommand "toolchain-check"
            {
              nativeBuildInputs = toolchain pkgs ++ lib.attrValues (wrappers pkgs);
            }
            ''
              for verb in ${lib.escapeShellArgs (lib.attrNames (commands pkgs))}; do
                command -v "dev-$verb" > /dev/null || {
                  echo "dev-$verb is not on PATH" >&2
                  exit 1
                }
              done
              touch "$out"
            '';

        # The build sandbox is an ideal stand-in for "some unrelated directory":
        # no git repo, no config, and no Python in it but what we plant here.
        #
        # This check exists because the flake shipped with exactly the opposite
        # behaviour. Every command ended in a bare "$@", so given no arguments
        # they acted on the CALLER's cwd: `nix run <url>#lint` -- the form CI and
        # a cold agent use -- printed "All checks passed!" having inspected none
        # of this repo, and `nix run <url>#fmt` rewrote source files outside the
        # repo entirely. Both are regressions a human reviewer will not notice,
        # so they get a machine.
        anchoring =
          pkgs.runCommand "anchoring-check"
            {
              nativeBuildInputs = lib.attrValues (wrappers pkgs);
            }
            ''
              decoy="$NIX_BUILD_TOP/decoy"
              logs="$NIX_BUILD_TOP/logs"
              mkdir -p "$decoy" "$logs"
              printf 'import os,sys\nx=1\n' > "$decoy/decoy.py"
              cp "$decoy/decoy.py" "$decoy/decoy.py.orig"
              cd "$decoy"

              # Read-only verbs must inspect this repo wherever they are called
              # from. Asserted through --show-files rather than through findings,
              # so this check does not start lying the day someone fixes the last
              # ruff warning.
              dev-lint --show-files > "$logs/files.log"
              grep -q '/main.py$' "$logs/files.log" || {
                echo "dev-lint did not look at the repo:" >&2
                cat "$logs/files.log" >&2
                exit 1
              }
              if grep -q decoy "$logs/files.log"; then
                echo "dev-lint reached into the caller's directory:" >&2
                cat "$logs/files.log" >&2
                exit 1
              fi

              # Verbs that write must refuse when there is no checkout, rather
              # than improvise one out of $PWD.
              for verb in fmt setup run; do
                if "dev-$verb" > "$logs/$verb.log" 2>&1; then
                  echo "dev-$verb should have refused outside a work tree:" >&2
                  cat "$logs/$verb.log" >&2
                  exit 1
                fi
              done

              # Nothing whatsoever may have appeared next to the caller: not a
              # reformatted file, not a .venv, not even a .ruff_cache.
              cmp "$decoy/decoy.py" "$decoy/decoy.py.orig"
              [ "$(find "$decoy" -mindepth 1 | wc -l)" -eq 2 ] || {
                echo "something was written into the caller's directory:" >&2
                find "$decoy" -mindepth 1 >&2
                exit 1
              }
              touch "$out"
            '';
      });

      # `nix fmt` -- formats the *Nix* in this repo; project code is `dev-fmt`.
      # nixfmt-tree (the treefmt wrapper) rather than bare nixfmt, because bare
      # nixfmt tries to parse every path handed to it and fails on non-Nix files.
      formatter = forAllSystems (pkgs: pkgs.nixfmt-tree);
    };
}
