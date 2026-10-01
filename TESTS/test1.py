class BankAccount:
    def __init__(self, balance):
        if balance < 0:
            raise ValueError("blance just > 0 not negatif")
        self.balance = balance
        self.__pin = "1234"

    def show_balance(self):
        self._log()
        print(f"Balance: {self.balance}")

    def _log(self):
        print("Checking balance...")

    def __check_pin(self, pin):
        return pin == self.__pin

    def withdraw(self, amount, pin):
        if self.__check_pin(pin):
            self.balance -= amount
            print(f"Withdrawn: {amount}")
        else:
            print("Wrong PIN")

account = BankAccount(-1)

account.show_balance()
account.withdraw(200, "1234")
account.withdraw(200, "1234")