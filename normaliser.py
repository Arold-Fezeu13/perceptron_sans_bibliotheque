def norm(liste=[]):
    max=[]
    for i in liste[0]:
        max.append(i)
    for line in liste:
        for i in range(len(liste[0])):
            if line[i]>max[i]:
                max[i]=line[i]
                
 
    for line in liste:
        for i in range(len(liste[0])):
            if max[i]!=0:
                line[i]/=(max[i])
            
    
    return liste 
