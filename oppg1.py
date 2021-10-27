
#Midpoint rule

b_1 = 0
b_2 = 1
c_1 = 0
c_2 = 1/2

i_1 = b_1 + b_2
i_2 = b_1 * c_1 + b_2 * c_2
i_3 = b_1  * c_1**2 + b_2 * c_2**2

print("Midpoint rule:")
print("1:",i_1)
print("2:",i_2)
print("3:",i_3)
print("The order of convergence when using the midpoint rule is therefore 2.")

#SSPRK3

b_1 = 1/6
b_2 = 1/6
b_3 = 2/3
c_1 = 0
c_2 = 1
c_3 = 1/2

p_1 = b_1 + b_2 + b_3
p_2 = b_1 * c_1 + b_2 * c_2 + b_3 * c_3
p_3 = b_1 * c_1**2 + b_2 * c_2**2 + b_3 * c_3**2

print("SSPRK3:")
print("1:",p_1)
print("2:",p_2)
print("3:",p_3)

# We will only look at a_21, a_31 and a_32 because all others are equal to 0

a_21 = 1
a_31 = 1/4
a_32 = 1/4

sum_a_b_c = a_21 * b_2 * c_1 + a_31 * b_3 * c_1 + a_32 * b_3 * c_2
print("Sum of products of as, bs and cs:", sum_a_b_c)
print("The order of convergence when using SSPRK3 is therefore 3.")