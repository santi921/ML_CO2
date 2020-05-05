import selfies
# worked in python 3

test_molecule1='CN1C(=O)C2=C(c3cc4c(s3)-c3sc(-c5ncc(C#N)s5)cc3C43OCCO3)N(C)C(=O)C2=C1c1cc2c(s1)-c1sc(-c3ncc(C#N)s3)cc1C21OCCO1' # non-fullerene acceptors for organic solar cells
selfies1=selfies.encoder(test_molecule1)
smiles1=selfies.decoder(selfies1)

print('test_molecule1: '+test_molecule1+'\n')
print('selfies1: '+selfies1+'\n')
print('smiles1: '+smiles1+'\n')
print('equal: '+str(test_molecule1==smiles1)+'\n\n\n')

my_alphabet=selfies.selfies_alphabet() # contains all semantically valid SELFIES symbols.


print(my_alphabet)


#this works for selfies --> selfies
#todo: make a selfies converter in the help finder
#vae works, also uses a selfie encoder