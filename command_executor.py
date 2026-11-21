import atexit
import codecs
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

import psutil


# ANSI escape sequences (CSI, OSC, single-char Fe). Stripped before display
# because the UI is a plain-text log, not a real terminal.
_ANSI_RE = re.compile(
    r'\x1b\[[0-?]*[ -/]*[@-~]'
    r'|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)'
    r'|\x1b[@-Z\\-_]'
)

# Injected via PYTHONPATH so the child believes stdout/stderr are terminals.
# Without this, tqdm (Lightning's default progress bar) auto-disables when
# its file object isn't a TTY, so we'd get no progress output at all.
_SITECUSTOMIZE = '''\
import sys as _sys
try:
    _sys.stdout.isatty = lambda: True
except Exception:
    pass
try:
    _sys.stderr.isatty = lambda: True
except Exception:
    pass
'''


class CommandExecutor:
    def __init__(self, on_output=None, on_finish=None):
        self.process = None
        self.on_output = on_output
        self.on_finish = on_finish
        self._reader_thread = None
        self._fake_tty_dir = None

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def _ensure_fake_tty_dir(self):
        if self._fake_tty_dir is not None:
            return self._fake_tty_dir
        d = tempfile.mkdtemp(prefix='vst_fake_tty_')
        try:
            with open(os.path.join(d, 'sitecustomize.py'), 'w', encoding='utf-8') as f:
                f.write(_SITECUSTOMIZE)
        except Exception as e:
            print(f'Warning: could not write sitecustomize.py: {e}')
        atexit.register(shutil.rmtree, d, ignore_errors=True)
        self._fake_tty_dir = d
        return d

    def execute_command(self, run_cmd):
        if self.is_running():
            print('The command is already running. Please wait for it to finish.')
            return False

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        env.setdefault('TERM', 'xterm-256color')
        try:
            fake_tty_dir = self._ensure_fake_tty_dir()
            prev = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = fake_tty_dir + (os.pathsep + prev if prev else '')
        except Exception as e:
            print(f'Warning: could not prepare fake-TTY environment: {e}')

        # Binary, unbuffered: FileIO.read(n) returns whatever is available,
        # which lets us process \r/\n separators as they arrive.
        self.process = subprocess.Popen(
            run_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=env,
        )

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        return True

    def _emit(self, text, mode):
        clean = _ANSI_RE.sub('', text)
        clean = clean.replace('\x07', '').replace('\x08', '')
        if mode == 'carriage' and not clean:
            return  # bare \r or an ANSI-only redraw — nothing to show
        if self.on_output:
            try:
                self.on_output(clean, mode)
            except Exception:
                pass

    def _reader_loop(self):
        try:
            raw = self.process.stdout if self.process else None
            if raw is not None:
                # Incremental decoder handles multi-byte UTF-8 split across reads.
                decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                pending = []
                pending_cr = False  # saw a bare \r; deciding \r\n vs \r

                def flush(mode):
                    if pending:
                        text = ''.join(pending)
                        pending.clear()
                        self._emit(text, mode)

                while True:
                    chunk = raw.read(4096)
                    if not chunk:
                        # EOF: flush any trailing decoder bytes, then pending.
                        tail = decoder.decode(b'', final=True)
                        for ch in tail:
                            if pending_cr:
                                pending_cr = False
                                if ch != '\n':
                                    flush('carriage')
                            if ch == '\r':
                                pending_cr = True
                            elif ch == '\n':
                                flush('newline')
                            else:
                                pending.append(ch)
                        if pending_cr:
                            flush('carriage')
                        else:
                            flush('newline')
                        break

                    text = decoder.decode(chunk, final=False)
                    for ch in text:
                        if pending_cr:
                            pending_cr = False
                            if ch == '\n':
                                flush('newline')     # \r\n → normal line
                                continue
                            else:
                                flush('carriage')    # \r + char → in-place update
                        if ch == '\r':
                            pending_cr = True
                        elif ch == '\n':
                            flush('newline')
                        else:
                            pending.append(ch)
        finally:
            rc = None
            try:
                rc = self.process.wait()
            except Exception:
                pass
            if self.on_finish:
                try:
                    self.on_finish(rc)
                except Exception:
                    pass

    def kill_command(self):
        if self.process and self.process.poll() is None:
            try:
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                print('The running process has been terminated.')
                return True
            except psutil.NoSuchProcess:
                print('The process does not exist.')
            except Exception as e:
                print(f'Error when terminating process: {e}')
        else:
            print('There is no running process to kill.')
        return False