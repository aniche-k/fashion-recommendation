import os
import numpy as np
import random
import glob
pants_arr=np.load("data_pants_common.npz")
sports_arr=np.load("data_pants_sports_common.npz")
party_arr=np.load("data_pants_party_common.npz")
office_arr=np.load("data_pants_office_common.npz")
pant_shirt=pants_arr["x"]
pant_shoe=pants_arr["y"]

colors=['BLACK','BLUE',
 'BROWN',
 'GREEN',
 'GREY',
 'KHAKI',
 'MARRON',
 'ORANGE',
 'PINK',
 'RED',
 'WHITE',
 'YELLOW']
index_pant=[]
index_shirt=[]
index_shoe=[]
l=[]
res=[]

# def recommend_default_new(occasion,uname):
#     path_pants = []
#     path_shirts = []
#     path_shoes = []
#     return_imgs = lambda item_type: glob.glob(os.path.join("users",uname,item_type,"**","*.jpg"),recursive=True)
#     pant_imgs = return_imgs("pants")
#     shirt_imgs = return_imgs("shirt") + return_imgs("t-shirt")
#     shoe_imgs = return_imgs("shoes")
#     empty_dict = {}
#     for shirt in shirt_imgs:
#         for pant in pant_imgs:
#             for shoe in shoe_imgs:
#                 empty_dict[shirt+pant+shoe] = "value"


#     print(shirt_imgs)




# recommend_default_new("test","hello")


def recommend_default(type,color,occasion,uname):
        max=float('-inf')
        #occasion="None"
        print(uname)
        if occasion == "sports":
            pant_shirt=sports_arr["x"]
        if occasion == "party":
            pant_shirt=party_arr["x"]
        if occasion == "office":
            pant_shirt=office_arr["x"]
    
        path_pant=os.listdir(r"users\{}\pants".format(uname))
        if(occasion=="office"):
            path_shirt=os.listdir(r"users\{}\shirt".format(uname))
        if(occasion=="sports"):
            path_shirt=os.listdir(r"users\{}\t-shirt".format(uname))
        if(occasion=="party"):
            path_shirt=os.listdir(r"users\{}\party".format(uname))
        path_shoe=os.listdir(r"users\{}\shoes".format(uname))
        
        # print(path_pant)
        pant_colors_available=[]
        shirt_colors_available=[]
        shoe_colors_available=[]
        if type != "pants":
            for i in path_pant:
                pant_colors_available.append(colors.index(i.upper()))
        if type != "shirt" and type != "t-shirt" :
            for i in path_shirt:
                shirt_colors_available.append(colors.index(i.upper()))
        if type != "shoes" :
            for i in path_shoe:
                shoe_colors_available.append(colors.index(i.upper()))
        if type == "pants":
            pant_colors_available.append(colors.index(color.upper()))
        if type == "shirt" or type == "t-shirt" :
            shirt_colors_available.append(colors.index(color.upper()))
        if type == "shoes":
            shoe_colors_available.append(colors.index(color.upper()))
        
        # print(pant_colors_available)
        for i in pant_colors_available:
            for j in shirt_colors_available:
                for k in shoe_colors_available:
                    if (i in index_pant and j in index_shirt and k in index_shoe):
                        pass
                    else:
                        if max < (0.5*pant_shirt[i][j] + 0.5*pant_shoe[i][k]):
                            max=(0.5*pant_shirt[i][j] + 0.5*pant_shoe[i][k])
                            pant_color=colors[i]
                            shirt_color=colors[j]
                            shoe_color=colors[k]
                            r=i
                            c=j
                            m=k
        index_pant.append(r)
        index_shirt.append(c)
        index_shoe.append(m)
                    
        l.append([pant_color,shirt_color,shoe_color])
                    
        print(l)
        

def select_image(l,type,path,uname,occasion):
    print("uname is: ",uname)
    pant_col=l[0].lower()
    shirt_col=l[1].lower()
    shoe_col=l[2].lower()
    path_pant=r"users\{}\pants\{}".format(uname,pant_col)
    if(occasion=="office"):
        path_shirt=r"users\{}\shirt\{}".format(uname,shirt_col)
    if(occasion=="sports"):
        path_shirt=r"users\{}\t-shirt\{}".format(uname,shirt_col)
    if(occasion=="party"):
        path_shirt=r"users\{}\party\{}".format(uname,shirt_col)
    #path_shirt=r"users\{}\shirt\{}".format(uname,shirt_col)
    path_shoes=r"users\{}\shoes\{}".format(uname,shoe_col)
    pants_list=os.listdir(path_pant)
    shirts_list=os.listdir(path_shirt)
    shoes_list=os.listdir(path_shoes)
    #i=int((pants_list[len(pants_list)-1]).rstrip(".jpg"))
    print("Pant list is:",os.listdir(path_pant))
    if type != "pants":
        pant_full_path=((path_pant+"\{}".format(random.choice(pants_list))).replace("\\","/")).replace("users/","")
    if type != "shirt" and type != "t-shirt" :
        shirt_full_path=(path_shirt+"\{}".format(random.choice(shirts_list))).replace("\\","/").replace("users/","")
    if type != "shoes":
        shoe_full_path=(path_shoes+"\{}".format(random.choice(shoes_list))).replace("\\","/").replace("users/","")
    if type == "pants":
        pant_full_path=path
    if type == "shirt" or type == "t-shirt":
        shirt_full_path=path
    if type == "shoes":
        shoe_full_path=path
    res.append([shirt_full_path,pant_full_path,shoe_full_path])

def execute(type,color,path,occasion,uname):
    count=0
    while(True):
        try:
            recommend_default(type,color,occasion,uname)
        except Exception as e :
            print("Count Is ",count)
            #print('Failed to do something: ' + str(e))
            break
        count+=1

    #print(l)
    #print(index_pant,index_shirt,index_shoe)       
    for i in range(count):
        select_image(l[i],type,path,uname,occasion)
    # print(res)
    return res

def clear():
    res.clear()
    l.clear()
    index_pant.clear()
    index_shirt.clear()
    index_shoe.clear()