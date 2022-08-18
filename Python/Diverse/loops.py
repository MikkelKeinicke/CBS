print("while loop:")
i = 0
while(i<10):
    print(i)
    i = i + 1

print("\nfor loop:")
for i in range(10):
    print(i)


print("\nfor loop with list:")
list = [2,3,5,3,2,3,425,4,5]

for i in list:
    print(i*2)


for i in list:
    if(i == 425):
        print(i)
    elif(i == 5):
        print(i)
    else:
        print(i*10)