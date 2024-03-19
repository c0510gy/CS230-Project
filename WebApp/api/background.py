import sys
import subprocess 
import threading  

try:
    import Queue as queue
except ImportError:
    import queue 

STDOUT = subprocess.STDOUT
PIPE = subprocess.PIPE
    
class AsyncOutputReader(threading.Thread):
    """ AsyncOutputReader: Asynchronous buffer reader.
    Implements and Asynchronous buffer reader that spawns a thread that will
    save all what is written in a buffer that can be store.
    Implements no locking readlines() and echo all the data to another descriptor.
    
    To do so, it uses a queue for all the lines.
    Arguments:
      - output: The incoming output from where we will read.
      - output_copy_fd: File descriptorto copy all the output
      - readline_callback: optional callback that will get the lines
    """
    def __init__(self, output, output_copy_fd, readline_callback=None):
        buffer = list()
        threading.Thread.__init__(self)
        self.buf = buffer
        self.output = output 
        self.output_copy_fd = output_copy_fd
        if readline_callback:
            self.readline_callback = readline_callback
        self.queue = queue.Queue()
        self.daemon = True # thread dies with the program
        self.start()

    """
    The thread will just read each line until the descriptor is closed.
    It inserts the inserted lines
    """
    def run(self):
        for line in iter(self.output.readline, ''):
            self.queue.put(line)
            self.readline_callback(line)
            if self.output_copy_fd: 
                self.output_copy_fd.write(line.decode("utf-8"))
        self.exit_callback()
        self.output.close()
    
	
    def readline_callback(self, line):
        pass
    
    def exit_callback(self):
        pass
    
    """
    Try to read a line. It saves it in a buffer.
    Returns  '' if none is available (a empty line has at least '\n')
    """
    def readline(self):
        try:
            line = self.queue.get_nowait()    
        except queue.Empty:
            return ''
        else:
            self.buf.append(line)
            return line
      
    """
    Read a bunch of lines.
    Returns [] if none is available
    """
    def readlines(self):
        lines = []
        while 1:
            line = self.readline()
            if line == '': break
            else: lines.append(line)
        return lines
        
    """
    Returns the buffer
    """
    def buffer(self):
        self.readlines()
        return self.buf
    

class BackgroundProcess(subprocess.Popen):
    ON_POSIX = 'posix' in sys.builtin_module_names
        
    """
    Executes a program in background, storing all the output in a asyncbuffer.
    
    Is the same than subprocess.Popen but:
    - stderr: None to ignore stderr, PIPE to have a 
          separated buffer or STDOUT to use the same 
          buffer than stdout. Default: STDOUT 
    - stdout_fd: Where the output will be copied.
    - stderr_fd: Where the output will be copied.
    - readline_callback: Callback called for each line read
    """
    def __init__ (self,  args, cwd=None, stderr=STDOUT, stdin=None, 
                  stdout_fd=sys.stdout, stderr_fd=sys.stderr, readline_callback=None):
        subprocess.Popen.__init__(self, args, cwd=cwd, stdout=PIPE, stderr=stderr, stdin=stdin, bufsize=1, close_fds=self.ON_POSIX, shell=True)
        self.async_stdout = AsyncOutputReader(self.stdout, stdout_fd, readline_callback)
        if stderr == PIPE:
            self.async_stderr = AsyncOutputReader(self.stderr, stderr_fd, readline_callback)
        elif stderr == STDOUT:
            self.async_stderr = self.async_stdout   
