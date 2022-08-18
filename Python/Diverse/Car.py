class Car:
    def __init__(self, brand):
        self.wheelsamount = 4
        self.brand = brand

    def remove_wheels(self, x):
        self.wheelsamount = self.wheelsamount - x

    def __str__(self):
        return self.brand + " " + str(self.wheelsamount)
