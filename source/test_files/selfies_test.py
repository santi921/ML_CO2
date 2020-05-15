
from helpers import xyz_to_smiles
from selfies import encoder, decoder, selfies_alphabet


# worked in python 3
fid = 0
temp = xyz_to_smiles()
for i in temp:
    selfies1 = encoder(i)
    smiles1 = decoder(selfies1)
    if (i == smiles1):
        fid += 1
    #print('equal: ' + str(i == smiles1) )
print("Selfies recovery fidelity: "+ str(fid/float(len(temp))))
# contains all semantically valid SELFIES segments
my_alphabet=selfies_alphabet()
#print(my_alphabet)


