import threading
import subprocess

def speak(message, callback):
    """
    Runs the given args in a subprocess.Popen, and then calls the function
    on_exit when the subprocess completes.
    on_exit is a callable object, and popen_args is a list/tuple of args that
    would give to subprocess.Popen.
    """
    def run_in_thread(callback, message):
        proc = subprocess.Popen(["say", "\"{}\"".format(message)])
        proc.wait()
        callback()
        return
    thread = threading.Thread(target=run_in_thread, args=(callback, message))
    thread.start()
    # returns immediately after the thread starts
    return thread
