"""Repair conflicts observed in the user's temporary Colab runtime.

Removes unused Gradio/Diffusers installations, keeps the project HF Hub pin,
and aligns gcsfs with datasets' fsspec version. Does not touch model caches.
"""
import subprocess
import sys


def run(*args):
    result = subprocess.run([sys.executable, '-m', 'pip', *args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    print('This project does not use gradio or diffusers; removing them from this Colab runtime.')
    run('uninstall', '-y', 'gradio', 'diffusers')
    run('install', '-r', 'requirements.txt', 'jedi==0.19.2',
        'gcsfs==2025.9.0', 'fsspec==2025.9.0')
    run('check')
    print('Dependency check passed. Restart the session, then rerun mount/configuration and import checks.')
