{
  # Keep this line accurate and one line long: `nix flake metadata` prints it,
  # and it is the first thing a cold agent learns about the repo.
  description = "cv-regie -- OpenCV/YOLO/DeepFace multi-camera director that picks and frames the best feed. Run `nix flake show` for the command map.";

  # nixpkgs is the only input, on purpose: the one thing flake-utils would buy
  # here -- eachDefaultSystem -- is the genAttrs in the canonical block below,
  # and a second lock node is a second upstream that can break.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    # `...` rather than a closed argument list. Measured on a throwaway flake:
    # `outputs = { nixpkgs }: ...` fails immediately with
    # `error: function 'outputs' called with unexpected argument 'self'`,
    # because nix passes `self` whether the flake asks for it or not, and every
    # further input is one more name the list would have to grow.
    #
    # `self` is mandatory, not decoration. It is the only way a wrapper sitting
    # in the store can name this repo's own files, and that is what anchors
    # every verb -- see rootPreamble in the canonical block. The price is that
    # each wrapper embeds the source store path, so every wrapper is rebuilt
    # when any tracked file changes; `dev-help` does not reference the source
    # and so is not.
    { self, nixpkgs, ... }:
    let
      lib = nixpkgs.lib;

      # Cosmetic -- the interactive dev-shell banner only. Matches the clone
      # directory and the GitHub repo name.
      repoName = "cv-regie";

      # ======================================================================
      # PER-REPO BLOCK 1 -- the toolchain
      # ======================================================================
      # Deliberately small: the heavy dependency set is pip's, not nix's. It
      # lives in requirements.txt and is installed into .venv by `dev-setup`.
      # Nix owns the interpreter and the launcher; pip owns the wheels. See
      # PER-REPO BLOCK 2 for why that split still needs help from nix on NixOS.
      #
      # python313, pinned by major rather than the rolling `python3` alias. A
      # venv records its interpreter as an absolute path -- the .venv this
      # flake builds has `home = /nix/store/...-python3-3.13.15/bin` in its
      # pyvenv.cfg -- so the day the alias moves to another store path, every
      # .venv built against it points at an interpreter that is no longer
      # there.
      #
      # Measured with the uv from this lock (uv 0.12.3):
      # `uv pip compile requirements.txt --python-version 3.13` resolves 107
      # packages, among them torch 2.13.0, tensorflow 2.21.0 and
      # opencv-python 5.0.0.93. Installing them gives the same 107.
      toolchain = pkgs: [
        # ---- invoked by a verb ----
        pkgs.uv # setup
        pkgs.ruff # lint, fmt

        # ---- not invoked by any verb ----
        # python313 is referenced by absolute store path in PER-REPO BLOCK 3
        # (UV_PYTHON), not through PATH; it is listed here for the human at the
        # prompt. git/jq/gnumake likewise: no command text below mentions them.
        pkgs.python313
        pkgs.git
        pkgs.jq
        pkgs.gnumake
      ];

      # ======================================================================
      # PER-REPO BLOCK 2 -- libraries that get dlopened, not linked
      # ======================================================================
      # The pip wheels load these by soname at import time and nothing on NixOS
      # puts them on the loader path. Each line below was measured against this
      # lock, in a venv holding numpy 2.5.2 and opencv-python 5.0.0.93, by
      # withholding the entry and reading the resulting ImportError:
      #
      #   import numpy needs   libstdc++.so.6        stdenv.cc.cc.lib
      #                        libz.so.1             zlib
      #   import cv2   needs   libxcb.so.1           libxcb
      #                        libGL.so.1            libGL
      #                        libgthread-2.0.so.0   glib
      #
      # The remaining four are not needed to import cv2 -- they are needed to
      # show a window, which is the point of this repo (cv.imshow, four call
      # sites in thread/leadThread.py). The wheel bundles its own Qt5 platform
      # plugin at cv2/qt/plugins/platforms/libqxcb.so, and `ldd` on that plugin
      # with the five entries above in place reports exactly these unresolved:
      #
      #                        libX11.so.6           libX11
      #                        libXext.so.6          libXext
      #                        libSM.so.6            libSM
      #                        libICE.so.6           libICE
      #
      # With all nine present that same `ldd` reports nothing unresolved and
      # `import cv2` succeeds.
      #
      # Do NOT "fix" this by adding pkgs.opencv4 or pkgs.ffmpeg. The wheel is
      # self-contained: cv2.getBuildInformation() reports "GUI: QT5",
      # "QT: YES (ver 5.15.19)" and "FFMPEG: YES", and the shared objects
      # behind them (libQt5Core-*.so.5.15.19, libavcodec-*.so.62, ...) ship
      # inside opencv_python.libs, and cv2.abi3.so finds them by itself:
      # `readelf -d` on it shows one RPATH, `$ORIGIN/../opencv_python.libs`.
      # A nixpkgs OpenCV beside that is a closure nothing loads.
      #
      # Linux-only attrs are safe here: the canonical ldPreamble forces
      # `nativeLibs pkgs` only on Linux, so aarch64-darwin never evaluates them.
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
      # Constants only, applied identically to the dev shell and to every
      # wrapper, so a verb cannot behave differently depending on how it was
      # invoked. Anything that must READ an existing value (LD_LIBRARY_PATH) or
      # UNSET something (SOURCE_DATE_EPOCH) is the canonical block's business,
      # not this attrset's.
      envVars =
        pkgs:
        {
          # Keep uv on the nix interpreter. Left alone it fetches its own
          # portable build instead -- measured: with both of these unset,
          # `uv venv --python 3.12` printed `Downloading
          # cpython-3.12.13-linux-x86_64-gnu (download) (32.6MiB)` and used
          # that. Two Pythons, one venv, and no way to tell which is live.
          UV_PYTHON = "${pkgs.python313}/bin/python";
          UV_PYTHON_DOWNLOADS = "never";
          # uv hardlinks from its CACHE into the venv, and when the two are on
          # different filesystems that fails. Measured with the cache on tmpfs
          # and the venv on btrfs: `warning: Failed to hardlink files; falling
          # back to full copy...` on the install, whose own third line tells
          # you to `export UV_LINK_MODE=copy` to suppress it. This is that.
          UV_LINK_MODE = "copy";
          # tensorflow arrives through tf-keras and deepface and logs a
          # CPU-feature probe at INFO on every import. Measured in the venv
          # this flake builds: with the variable unset, `import tensorflow`
          # prints `cpu_feature_guard.cc:227] This TensorFlow binary is
          # optimized to use available CPU instructions...`; with it at 1 that
          # line is gone. The absl `cudart_stub.cc:31` line prints either way,
          # because it is emitted before absl's logger is initialised -- this
          # variable cannot suppress it and does not claim to. 1 filters INFO
          # only; warnings and errors still print. Do not raise it to 2 or 3,
          # which starts hiding real problems.
          TF_CPP_MIN_LOG_LEVEL = "1";
        }
        // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          # The opencv-python wheel ships exactly one Qt platform plugin --
          # cv2/qt/plugins/platforms/ contains libqxcb.so and nothing else --
          # so xcb is the only choice there is. Naming it up front stops Qt
          # probing for a "wayland" plugin that is not in the wheel. Linux
          # only: the darwin wheel uses cocoa, where this value would break
          # imshow.
          QT_QPA_PLATFORM = "xcb";
        };

      # ======================================================================
      # PER-REPO BLOCK 4 -- the command map
      # ======================================================================
      # THE single source of truth: it generates `apps` (so `nix run .#lint`
      # works), the `dev-*` wrappers on PATH inside the shell, and `dev-help`.
      #
      # No `test` and no `build` verb: this repo has no test suite and no
      # packaging step, and a stub that echoed "not applicable" would turn
      # `nix flake show` into a liar. Absence is information.
      commands = pkgs: {
        setup = {
          # --allow-existing is not cosmetic. Verified against the uv in this
          # lock (0.12.3): a second `uv venv .venv` -- the obvious move after
          # editing requirements.txt -- exits 2 with "A virtual environment
          # already exists at: .venv" before the install line ever runs. Do not
          # take uv's own hint and reach for --clear instead: that deletes a
          # multi-gigabyte venv to add one package.
          description = "(network, ~7 GB of .venv: 107 packages, 16 of them nvidia CUDA) create or update .venv from requirements.txt";
          # A .venv belongs to a checkout. $REPO_ROOT can be the read-only
          # store snapshot, and unpacking the venv into whichever directory the
          # caller happened to be standing in is worse than refusing.
          text = ''
            need_writable_checkout
            uv venv --allow-existing "$REPO_ROOT/.venv"
            uv pip install --python "$REPO_ROOT/.venv/bin/python" -r "$REPO_ROOT/requirements.txt" "$@"
          '';
        };
        lint = {
          description = "ruff check over this repo, from any directory";
          # `cd` first, then a bare `.` default. Both halves are load-bearing:
          # `ruff check "$@"` alone checks the caller's cwd, and even
          # `ruff check "''${@:-$REPO_ROOT}"` still checks the cwd the moment
          # the caller passes a flag rather than a path (`--fix`,
          # `--select F401`), because any argument suppresses the default.
          # Standing in the root closes both, and it makes a relative path
          # argument mean the same thing from wherever it was typed.
          #
          # --no-cache unconditionally. Measured: `ruff check /elsewhere/src`
          # run from an unrelated directory puts .ruff_cache in THAT directory,
          # not beside the files it was given -- the cache follows the
          # process's cwd. So in a checkout it is a directory nobody asked for
          # (hence the .gitignore entry), and when $REPO_ROOT is the store
          # snapshot it is fatal: `error: Failed to initialize cache at
          # /nix/store/...-source/.ruff_cache: Read-only file system`,
          # `ruff failed`. Five Python files do not need an incremental cache.
          text = ''
            cd "$REPO_ROOT"
            ruff check --no-cache "''${@:-.}"
          '';
        };
        fmt = {
          description = "ruff format (rewrites files, so it needs a writable checkout)";
          # MUTATING, so it takes the guard and gets no $SRC_ROOT fallback:
          # $REPO_ROOT is the read-only store snapshot whenever the caller is
          # not standing in a checkout, and there is nothing useful to format
          # there. No cwd default either -- a `fmt` that falls back to the
          # caller's directory is a formatter pointed at a stranger's source,
          # which is what checks.verbAnchoring below exists to catch.
          text = ''
            need_writable_checkout
            cd "$REPO_ROOT"
            ruff format --no-cache "''${@:-.}"
          '';
        };
        run = {
          # main.py hardcodes ports [0, 1] and resolves both
          # "model/pose/yolov8n-pose.pt" and "./database" against the CURRENT
          # directory rather than against the script. That is why this cds to
          # the root instead of merely naming main.py absolutely: started
          # anywhere else it writes the pose weights beside the caller. Neither
          # path is tracked here, and a first run was watched doing exactly
          # that -- ultralytics fetched yolov8n-pose.pt (6.5 MB) from its
          # GitHub assets release into $REPO_ROOT/model/pose/, and cv2 reported
          # `VIDEOIO(V4L2:/dev/video1): can't open camera by index` on the
          # second port. `*.pt` is already in .gitignore.
          #
          # The venv interpreter by absolute path, not a bare `python`: the
          # wrappers prepend the nix toolchain to PATH, so a bare name resolves
          # to the store copy and misses everything `setup` installed.
          description = "(network on first run: downloads the YOLO pose weights; needs webcams on ports 0 and 1 and opens OpenCV windows) start the director";
          text = ''
            need_writable_checkout
            if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
              echo "''${0##*/}: no .venv in $REPO_ROOT -- run dev-setup first." >&2
              exit 1
            fi
            cd "$REPO_ROOT"
            "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/main.py" "$@"
          '';
        };
      };

      # ======================================================================
      # PER-REPO BLOCK 5 -- checks that know what this repo does
      # ======================================================================
      # The canonical `anchoring` check proves the MECHANISM behaves: that
      # rootPreamble refuses a foreign tree and adopts a real checkout. It
      # cannot prove that THIS repo's verbs use it. This one drives the real
      # wrappers from inside a decoy and asserts exactly three things: that a
      # mutating verb does not SUCCEED there, that the read-only verb grades
      # this repo and not the decoy, and that nothing at all is written beside
      # the caller. A verb that starts pointing at $PWD fails at least one of
      # them.
      extraChecks = pkgs: {
        verbAnchoring =
          pkgs.runCommand "verb-anchoring-check"
            {
              nativeBuildInputs = lib.attrValues (wrappers pkgs);
            }
            ''
              set -euo pipefail

              # The build sandbox is an ideal stand-in for "some unrelated
              # directory": no git repo, no config, and no Python source in it
              # but what is planted here. The decoy carries a flake.nix that
              # differs, which is what the canonical anchor compares against.
              mkdir decoy
              cd decoy
              printf 'import os,sys\nx  =1\n' > decoy_only.py
              printf '{\n  description = "a different repo";\n  outputs = _: { };\n}\n' > flake.nix
              cp -r . ../decoy.orig

              # Read-only verb: it must grade THIS repo from anywhere, and
              # never the caller's directory. Asserted through --show-files
              # rather than through findings, so the check does not start lying
              # the day somebody fixes the last ruff warning. Grepping for a
              # filename this repo does not contain, rather than for the decoy
              # directory: a leaked tool standing in the decoy may print bare
              # relative paths, and then a grep for "decoy" matches nothing.
              dev-lint --show-files > files.log
              grep -q '/main.py$' files.log || {
                echo "dev-lint did not look at this repo" >&2
                cat files.log >&2
                exit 1
              }
              if grep -q decoy_only files.log; then
                echo "dev-lint graded the caller's directory" >&2
                cat files.log >&2
                exit 1
              fi

              # Mutating verbs: refusal, not silence, and no improvising a
              # target out of $PWD.
              for verb in setup fmt run; do
                if "dev-$verb" > "$verb.log" 2>&1; then
                  echo "dev-$verb should have refused outside a checkout" >&2
                  cat "$verb.log" >&2
                  exit 1
                fi
              done

              # Nothing may have appeared beside the caller: not a reformatted
              # file, not a .venv, not a .ruff_cache. `*.log`, and every log
              # this check writes must match it -- a file named plainly `log`
              # would not be excluded and would fail the diff.
              diff -r --exclude='*.log' . ../decoy.orig
              touch "$out"
            '';
      };

      # >>>>> BEGIN CANONICAL MACHINERY v1 <<<<<
      # ======================================================================
      # Everything from the BEGIN sentinel above to the END sentinel on the last
      # line of this file is fleet-canonical text: the same bytes in every repo
      # that carries this flake style. That is a checkable claim, not a boast --
      #
      #   sed -n '/BEGIN CANONICAL MACHINERY v1/,$p' flake.nix | sha256sum
      #
      # prints the same digest in every repo, or one of them has been edited.
      # (`,$p`, not a range ending on the END sentinel: a range whose closing
      # pattern were spelled out here would terminate on this very comment.)
      # Nothing here names a repository, a language, a tool or a project file.
      # If you find such a name below, it is contamination: the fix is to move
      # it into the per-repo section above, never to special-case it here.
      #
      # This region READS exactly these names from the per-repo section:
      #   nixpkgs  self  lib  repoName  toolchain  nativeLibs  envVars
      #   commands  extraChecks
      # and DEFINES exactly these:
      #   systems  forAllSystems  ldPreamble  rootPreamble  guardPreamble
      #   wrappers  helpFor  anchorCheck
      # plus the four flake outputs apps / devShells / checks / formatter.
      # Anything else in scope is invisible to it. The types of those eight
      # inputs, and the shell variables this region exports into command texts,
      # are specified in INTERFACE.md, which travels with this block.
      #
      # To change behaviour here you change it in every repo at once and bump
      # the version in both sentinels. A local edit is a bug by construction:
      # the digest above stops matching, and -- because rootPreamble anchors on
      # flake.nix byte-identity -- an edited working tree also stops being
      # recognised by wrappers built from the previous revision.
      # ======================================================================

      # ---- systems policy: decided once for the whole fleet ----
      #
      # Read this list as "evaluated on three, built on one". That is what was
      # measured, and it is all it means:
      #   * `nix flake check --all-systems` passes, so every output attribute
      #     below EVALUATES on all three systems.
      #   * only x86_64-linux has ever been BUILT. The machine this was verified
      #     on has no aarch64 emulation -- no binfmt handler, and `extra-
      #     platforms` is x86-only -- so aarch64 cannot be built there at all.
      # It is not a statement that anything works on aarch64. Do not upgrade it
      # into one in a README.
      #
      # Evaluating all three is still worth its seconds, because the failure it
      # catches is an eval-time failure: a `pkgs.<attr>` that exists on Linux
      # and not on darwin (`stdenv.cc.cc.lib` is the usual one) throws during
      # evaluation, and `nix flake check` without --all-systems checks only the
      # current system and sails straight past it.
      #
      # x86_64-darwin is deliberately absent. nixpkgs 26.11 replaced that whole
      # attribute set with a `throw`. genAttrs is lazy, so plain `nix develop`
      # on Linux would not notice -- it detonates later, on the --all-systems
      # run this policy requires. Add it back only against a separate
      # nixpkgs-26.05-darwin input.
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      # Stand-in for flake-utils.lib.eachDefaultSystem. Passes `pkgs` rather
      # than a system string, because that is what every call site wants, and
      # keeps the system list in this file rather than in a second input's
      # hardcoded copy of it.
      forAllSystems = f: lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      # Prepend, never assign: a host LD_LIBRARY_PATH may be carrying something
      # the user needs, and clobbering it breaks binaries they launch from here.
      # Linux only -- on darwin the loader variable is DYLD_*, and exporting a
      # Linux-shaped value there is at best useless.
      #
      # `&&` short-circuits in Nix, so on darwin `nativeLibs pkgs` is never
      # forced. That is load-bearing for the systems policy above: it is what
      # lets a repo list Linux-only attrs in nativeLibs and still evaluate on
      # aarch64-darwin. Do not reorder the two operands.
      ldPreamble =
        pkgs:
        lib.optionalString (pkgs.stdenv.hostPlatform.isLinux && nativeLibs pkgs != [ ]) ''
          export LD_LIBRARY_PATH="${lib.makeLibraryPath (nativeLibs pkgs)}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';

      # Every command gets $SRC_ROOT and $REPO_ROOT. `nix run` and `nix develop`
      # both start in whatever directory they were invoked from, and no verb may
      # act on that directory -- these two are what it acts on instead.
      #
      # $SRC_ROOT is this flake's own source, snapshotted into the store when
      # the flake was evaluated. It is the one anchor that is always available:
      # `nix run /path/to/repo#lint` tells the running program nothing whatever
      # about /path/to/repo (flake refs are location-independent by design, and
      # there is no $FLAKE_DIR to read), so without `self` a wrapper invoked
      # that way has literally no way to name the repo it belongs to. Two
      # limitations worth knowing: it is read-only, being a store path, and in a
      # git checkout it contains only TRACKED files.
      #
      # $REPO_ROOT is the writable checkout when the caller is standing in one,
      # and $SRC_ROOT when they are not. Three things this deliberately is NOT:
      #
      #   * NOT `pwd`. A fallback to the caller's directory is how `fmt`
      #     rewrites a stranger's source tree and how `lint` prints "all checks
      #     passed" having read none of this repo.
      #   * NOT `git rev-parse --show-toplevel`. Run from inside some OTHER git
      #     repo it cheerfully answers with THAT repo's top level. It also needs
      #     git on PATH and a .git directory, so it fails on an export and in
      #     any wrapper whose toolchain omits git.
      #   * NOT an inherited $REPO_ROOT from the environment. The dev shell
      #     EXPORTS this variable, so honouring it would mean that running
      #     `nix run /path/to/B#fmt` from inside repo A's dev shell points B's
      #     formatter at A. An explicit path argument is how a caller overrides
      #     a verb's target; an ambient variable is how they do it by accident.
      #
      # Instead: walk up from $PWD and take the first ancestor that IS this
      # repo, proved by carrying a byte-identical flake.nix. A single tracked
      # filename, a marker directory, or a set of them is not proof -- sibling
      # repos in a fleet share those, and a decoy can be built to carry any list
      # of names you care to publish. The whole flake.nix is what distinguishes
      # repos, because description, toolchain and command map all differ, so the
      # whole flake.nix is what gets compared. Compared with bash's own
      # `$(<file)` rather than cmp or sha256sum, so the check depends on no
      # package at all -- pure builtins, correct even in a wrapper whose PATH
      # carries nothing but the repo's own toolchain.
      #
      # Consequence worth knowing: edit flake.nix and the dev-* wrappers in an
      # already-open `nix develop` stop recognising the tree, because they were
      # built from the previous flake.nix. That is a stale shell telling you so
      # -- re-enter it. `nix run` re-evaluates every time and never sees this.
      rootPreamble = ''
        SRC_ROOT=${lib.escapeShellArg "${self}"}
        export SRC_ROOT

        _dev_find_root() {
          local dir ref
          ref=$(<"$SRC_ROOT/flake.nix") || return 1
          dir=$(
            unset CDPATH
            cd -P -- "''${1:-.}" 2>/dev/null && pwd
          ) || return 1
          while [ -n "$dir" ]; do
            if [ -f "$dir/flake.nix" ] && [ "$(<"$dir/flake.nix")" = "$ref" ]; then
              printf '%s\n' "$dir"
              return 0
            fi
            dir=''${dir%/*}
          done
          return 1
        }

        REPO_ROOT="$(_dev_find_root "$PWD" || printf '%s\n' "$SRC_ROOT")"
        export REPO_ROOT
      '';

      # Wrappers only, not the shellHook -- an interactive shell has no business
      # carrying this function around. Any command text that writes files calls
      # it first, and it is the reason a mutating verb can fail loudly instead
      # of falling back to "well, the cwd then".
      #
      # The test is $REPO_ROOT != $SRC_ROOT, i.e. "rootPreamble found a real
      # checkout", not a permission or a store-path-prefix test. Both of those
      # answer a narrower question: a checkout may be read-only for unrelated
      # reasons, and a store path is not the only tree we must refuse to write.
      guardPreamble = ''
        need_writable_checkout() {
          if [ "$REPO_ROOT" != "$SRC_ROOT" ]; then
            return 0
          fi
          echo "''${0##*/}: this command rewrites files, so it needs a writable" >&2
          echo "checkout of this repo -- and standing in $PWD there is none: no" >&2
          echo "parent directory carries this flake's flake.nix. The only tree in" >&2
          echo "reach is the read-only store snapshot $SRC_ROOT, and rewriting" >&2
          echo "$PWD instead is exactly the bug this guard exists to prevent." >&2
          echo "cd into the repo (or \`nix develop\` it), or pass an explicit path." >&2
          exit 1
        }
      '';

      # One derivation per command, reused by both `apps` and the dev shell, so
      # the two can never diverge. `dev-` prefixed because a bare `test` binary
      # earlier on PATH would shadow the POSIX shell builtin and quietly break
      # every script in the repo that uses it.
      #
      # writeShellApplication, not writeShellScriptBin: it runs shellcheck at
      # BUILD time and sets `set -euo pipefail`, so an unquoted $@ or a silently
      # ignored failure is a `nix flake check` failure rather than a surprise in
      # front of an agent.
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
              ${guardPreamble}
              ${ldPreamble pkgs}
              ${cmd.text}
            '';
          }
        ) (commands pkgs);

      # `dev-help` is generated from the same attrset as everything else, so it
      # cannot describe a verb that does not exist or miss one that does. No
      # runtimeInputs: printing the map must work with nothing installed.
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

      # The regression gate for rootPreamble and guardPreamble, which are the
      # two pieces of this flake that can silently damage a tree that is not
      # this repo. It tests the MECHANISM, not any verb, which is precisely what
      # makes it fleet-generic: it needs to know nothing about what this repo
      # does, only that the anchor resolves and the guard refuses.
      #
      # The decoy is a real directory carrying a real flake.nix that differs.
      # Marker-file anchors pass a decoy like this -- that is the whole point of
      # the probe -- and so does any anchor that trusts `pwd`. Probe 2 is the
      # other half, and without it a guard that refused everything would score a
      # perfect pass: a tree that IS byte-identical must still be adopted, or
      # every mutating verb in the repo is dead. Probe 3 pins the subdirectory
      # case, which is the normal one for an agent working inside a repo.
      #
      # A per-repo probe that drives the actual verbs is strictly better and
      # cannot live here -- it has to know which verb writes and which needs a
      # network. INTERFACE.md shows how to add one via `extraChecks`.
      anchorCheck =
        pkgs:
        pkgs.runCommand "anchor-check" { } ''
          set -euo pipefail

          # The two preambles under test, verbatim, in a file the probes source.
          # A quoted heredoc, so every $ below is the bash the wrappers see.
          cat > preamble.sh <<'CANONICAL_PREAMBLE_EOF'
          ${rootPreamble}
          ${guardPreamble}
          CANONICAL_PREAMBLE_EOF

          mkdir decoy
          printf '{\n  description = "a different repo";\n  outputs = _: { };\n}\n' > decoy/flake.nix
          printf 'do not touch me\n' > decoy/victim.txt
          cp -r decoy decoy.orig

          # ---- probe 1: a foreign tree must not be adopted ----
          if ! ( cd decoy && . ../preamble.sh && [ "$REPO_ROOT" = "$SRC_ROOT" ] ); then
            echo "anchor adopted a directory that is not this repo" >&2
            exit 1
          fi
          # In a subshell: need_writable_checkout ends in `exit`, which would
          # otherwise take this whole build down instead of failing a condition.
          if ( cd decoy && . ../preamble.sh && need_writable_checkout ) > guard.log 2>&1; then
            echo "need_writable_checkout accepted a tree that is not this repo" >&2
            exit 1
          fi
          if ! diff -r decoy decoy.orig; then
            echo "the probes modified the foreign tree" >&2
            exit 1
          fi

          # ---- probe 2: a byte-identical checkout must be adopted ----
          cp -r ${lib.escapeShellArg "${self}"} checkout
          chmod -R u+w checkout
          if ! ( cd checkout && . ../preamble.sh &&
                 [ "$REPO_ROOT" = "$(pwd -P)" ] && need_writable_checkout ); then
            echo "anchor refused a byte-identical checkout of this repo" >&2
            exit 1
          fi

          # ---- probe 3: from a subdirectory, still the checkout root ----
          mkdir -p checkout/probe3/deeper
          if ! ( cd checkout/probe3/deeper && . ../../../preamble.sh &&
                 [ "$REPO_ROOT" = "$(cd -P ../.. && pwd)" ] ); then
            echo "anchor did not walk up to the checkout root from a subdirectory" >&2
            exit 1
          fi

          touch "$out"
        '';
    in
    {
      # `nix flake show` -- the discovery entrypoint, and deliberately the whole
      # machine-facing contract: every app carries a meta.description, which
      # `nix flake show` prints inline and `nix flake show --json` exposes at
      # .apps.<system>.<name>.description. Pure evaluation, so an agent gets the
      # entire command map in one cheap call without reading a README.
      #
      # Do NOT invent a top-level output for this (`agentManifest`, `probeThing`
      # ...). Nix answers with `warning: unknown flake output '<name>'` on every
      # single `nix flake check`, forever.
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

          # Natively-compiled extension modules are routinely built at -O0,
          # where glibc's _FORTIFY_SOURCE stops being a warning and becomes a
          # hard error.
          hardeningDisable = [ "fortify" ];

          shellHook = ''
            # mkShell inherits SOURCE_DATE_EPOCH=315532800 (1980-01-01) from
            # stdenv, and any wheel or zip built in here then dies with "ZIP does
            # not support timestamps before 1980".
            unset SOURCE_DATE_EPOCH

            # $REPO_ROOT and $SRC_ROOT are exported here as a convenience for
            # the human at the prompt. Every wrapper re-resolves them from
            # scratch and none of them reads these, on purpose: a stale value
            # exported by one repo's shell must never steer another repo's verb.
            ${rootPreamble}
            ${ldPreamble pkgs}

            # Nothing networked, nothing stateful and nothing interactive above
            # this line, and nothing below it either. No environment
            # bootstrapping, no dependency installation, no `read`, no
            # `exec $SHELL`. Bootstrapping in the hook makes a cold
            # `nix develop -c <anything>` start downloading before it runs
            # anything, on EVERY invocation -- the exact failure an unattended
            # agent cannot diagnose. That is what a `setup` verb is for.

            # The banner is interactive-only, and this guard is load-bearing:
            # shellHook output lands on the STDOUT of `nix develop -c <cmd>`, so
            # an unguarded echo corrupts anything parsing it
            # (`nix develop -c cat x.json | jq` fails to parse). $- is the only
            # reliable discriminator here -- it lacks `i` for `nix develop -c`
            # and has it at an interactive prompt. Do not test $PS1 (unset in
            # both) or $IN_NIX_SHELL (set in both). >&2 is the second layer, for
            # the case where a caller runs us on a pty.
            case $- in
              *i*) echo "${repoName} dev shell -- 'dev-help' for the command map" >&2 ;;
            esac
          '';
        };
      });

      # `nix flake check` -- honest by construction, and the only gate this
      # style has. `toolchain` realises the whole toolchain closure (so a typo'd
      # or currently-broken attr fails here, not halfway through a task) and
      # builds every wrapper, which runs shellcheck over every command text.
      # `anchoring` is the regression test described above.
      #
      # Repo-specific checks go in `extraChecks`, never here. They may not
      # shadow either canonical name: silently replacing `anchoring` with
      # something weaker is the exact failure this whole file exists to make
      # impossible, so a collision is an eval error with both names in it.
      #
      # NEVER add a check that always passes. An agent reads "all checks
      # passed!" as a signal, and a fake check makes `nix flake check` a liar.
      checks = forAllSystems (
        pkgs:
        let
          canonical = {
            toolchain =
              pkgs.runCommand "toolchain-check"
                {
                  nativeBuildInputs = toolchain pkgs ++ lib.attrValues (wrappers pkgs) ++ [ (helpFor pkgs) ];
                }
                ''
                  set -euo pipefail
                  dev-help > help.txt

                  # A while-read over a heredoc rather than `for x in <list>`,
                  # which is a bash syntax error when the list is empty -- and a
                  # repo with no verbs yet is a legitimate state.
                  while IFS= read -r verb; do
                    [ -n "$verb" ] || continue
                    command -v "dev-$verb" > /dev/null || {
                      echo "dev-$verb is not on PATH" >&2
                      exit 1
                    }
                    grep -q -- "dev-$verb" help.txt || {
                      echo "dev-$verb is missing from the dev-help map" >&2
                      exit 1
                    }
                  done <<'CANONICAL_VERBS_EOF'
                  ${lib.concatStringsSep "\n" (lib.attrNames (commands pkgs))}
                  CANONICAL_VERBS_EOF

                  touch "$out"
                '';
            anchoring = anchorCheck pkgs;
          };
          extra = extraChecks pkgs;
          clash = lib.intersectLists (lib.attrNames canonical) (lib.attrNames extra);
        in
        if clash != [ ] then
          throw "extraChecks must not redefine canonical checks: ${lib.concatStringsSep ", " clash}"
        else
          canonical // extra
      );

      # `nix fmt` -- formats the *Nix* in this repo; project code gets a `fmt`
      # verb. nixfmt-tree (the treefmt wrapper) rather than bare nixfmt, because
      # bare nixfmt tries to parse every path handed to it and fails on non-Nix
      # files. This file ships already formatted, so `nix fmt` is a no-op rather
      # than a diff across the fleet.
      #
      # This is the one verb here NOT anchored to $REPO_ROOT, and it cannot be:
      # `nix fmt` is nix's own verb, and nix -- not this flake -- decides which
      # paths the formatter receives, passing the cwd when the user names none.
      # A wrapper that overrode them would break `nix fmt path/to/one/file.nix`,
      # and it cannot tell that "." apart from the default. So `nix fmt` formats
      # where you stand, by design; the `fmt` verb is the anchored one.
      formatter = forAllSystems (pkgs: pkgs.nixfmt-tree);
    };
}
# >>>>> END CANONICAL MACHINERY v1 <<<<<
