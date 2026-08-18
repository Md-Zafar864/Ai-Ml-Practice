# n=15
# sum=0
# for i in range(1,n+1):
#     if(i%2==0):
#         sum=sum+i

# print(sum) 
# 
#  
# 3. table
# n=3
# for i in range(1,10+1):
#     if(i==5):
#         continue
#     print(n*i)
# str="abcd"
# rev=""

# for i in str:
#     rev=i+rev

# print(rev)



# def fun(a,b):
#     return a+b
# print(fun(3,5))    

def check(v):
    if(v%2==0):
        return True
    else:
        return False
    
v=int(input("Enter Number: "))
print(check(v))