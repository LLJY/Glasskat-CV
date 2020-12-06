class RingBuffer:
    def __init__(self, max_size):
        self.max_Size = max_size
        self.data = []
        # set the current index to 0
        self.cur = 0
        self.isFull = False

    def append(self, newData):
        if (self.isFull):
            self.data[self.cur] = newData
            # change the current to zero for the next append to maintain the ring buffer
            if self.cur == self.max_Size:
                self.cur = 0
            else:
                self.cur += 1
        else:
            # if the list is not full yet, just append the data
            self.data.append(newData)
            self.isFull = (len(self.data) == self.max_Size)

    # Flushes all the data in the ringbuffer
    def flush(self):
        self.isFull = False
        self.data = []
        # for JVM, trigger GC here to avoid taking up a lot of ram

    # cur is the current
    def next(self, idx):
        # if the distance from the current index
        if idx >= (self.max_Size - self.cur):
            return self.data[(idx + self.cur) - self.max_Size]
        return self.data[self.cur + idx]
