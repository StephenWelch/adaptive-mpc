import time
from typing import Optional, Callable, Any, TypeVar, Union, Generic

T = TypeVar('T')

class Timer:
    """Context manager for timing code execution.
    
    Usage:
        with Timer() as t:
            # code to time
            pass
        print(f"Code took {t.duration:.2f} seconds")
        
        # With callback
        def log_time(duration: float):
            print(f"Operation took {duration:.2f} seconds")
            
        with Timer(callback=log_time):
            # code to time
            pass
    """
    
    def __init__(self, callback: Optional[Callable[[float], None]] = None):
        """Initialize the timer.
        
        Args:
            callback: Optional function to call with the duration when the context exits.
                     The function should accept a single float parameter (duration in seconds).
        """
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.callback = callback
        
    @property
    def duration(self) -> float:
        """Get the duration of the timed operation in seconds."""
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("Timer has not been used in a context manager")
        return self.end_time - self.start_time
        
    def __enter__(self):
        """Start timing when entering the context."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting the context and call the callback if provided."""
        self.end_time = time.time()
        if self.callback is not None:
            self.callback(self.duration)

class PeriodicCallback(Generic[T]):
    """Calls an underlying callback every n times this class is called.
    
    Usage:
        def my_callback(x: int) -> str:
            return f"Processed {x}"
            
        periodic = PeriodicCallback[str](my_callback, n=3)
        for i in range(5):
            result = periodic(i)  # Only calls my_callback when i is 2
    """
    
    def __init__(self, callback: Callable[..., T] | None = None, n: int = 1) -> None:
        """Initialize the periodic callback.
        
        Args:
            callback: The function to call periodically
            n: Call the function every n times this object is called
        """
        self.callback = callback
        self.n = n
        self.count = 0
        
    def __call__(self, *args: Any, **kwargs: Any) -> Optional[T]:
        """Call the underlying callback every n times.
        
        Args:
            *args: Positional arguments to pass to the callback
            **kwargs: Keyword arguments to pass to the callback
            
        Returns:
            The result of the callback if it was called, None otherwise
        """
        self.count += 1
        if self.count % self.n == 0 and self.callback is not None:
            return self.callback(*args, **kwargs)
        return None
