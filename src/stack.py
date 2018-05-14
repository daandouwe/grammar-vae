class Stack:
    def __init__(self, grammar, start_symbol):
        self.grammar = grammar
        self._stack = [start_symbol]

    def pop(self):
        return self._stack.pop()

    def push(self, symbol):
        self._stack.append(symbol)

    @property
    def empty(self):
        return not self._stack


if __name__ == '__main__':
    # Usage:
    from grammar import GCFG, S, T

    stack = Stack(grammar=GCFG, start_symbol=S)
    print(stack.nonempty)
