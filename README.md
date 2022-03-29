# MedPipe3DTutorial
Showing how to use MedPipe (including MedEye3d  MedEval3D  HDF5 and MONAI preprocessing)


# First example
Example is based on public dataset that can be found in link below

https://wiki.cancerimagingarchive.net/display/Public/CT-ORG%3A+CT+volumes+with+multiple+organ+segmentations

labels without readme file should be put in the labels folder and volumes in "volumes 0-49" folder - there should be the same number of labels as volumes - you do not need to download all of them if you do not want to. 


Next we establish imports
In case any of this packages are not already installed do it now

```
using Pkg
Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")

using MedEye3d
using Distributions
using Clustering
using IrrationalConstants
using ParallelStencil
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedPipe3D.visualizationFromHdf5, MedPipe3D.distinctColorsSaved
using CUDA
using HDF5
```

We define some constants

```

#representing number that is the patient id in this dataset
patienGroupName="0"
z=7# how big is the area from which we collect data to construct probability distributions
klusterNumb = 5# number of clusters - number of probability distributions we will use


#directory of folder with files in this directory all of the image files should be in subfolder volumes 0-49 and labels labels if one ill use lines below
data_dir = "D:\\dataSets\\CTORGmini\\"
#directory where we want to store our HDF5 that we will use
pathToHDF5="D:\\dataSets\\forMainHDF5\\smallLiverDataSet.hdf5"
```
Finding paths to files and constructing tuples with them - check weather they will point to the same patient

```
train_labels = map(fileEntry-> joinpath(data_dir,"labels",fileEntry),readdir(joinpath(data_dir,"labels"); sort=true))
train_images = map(fileEntry-> joinpath(data_dir,"volumes 0-49",fileEntry),readdir(joinpath(data_dir,"volumes 0-49"); sort=true))


#zipping so we will have tuples with image and label names
zipped= collect(zip(train_images,train_labels))
tupl=zipped[1]
tupl
```
for example in my case tuple of directories pointing to the patient 0 looks like below
("D:\\dataSets\\CTORGmini\\volumes 0-49\\volume-0.nii.gz", "D:\\dataSets\\CTORGmini\\labels\\labels-0.nii.gz")

Next we load and preprocess data - here I use defoult transformations - If You want to use some other transformations you can supply third argument to loadByMonaiFromImageAndLabelPaths function more in LoadFromMonai.loadByMonaiFromImageAndLabelPaths function doc
```
loaded = LoadFromMonai.loadByMonaiFromImageAndLabelPaths(tupl[1],tupl[2])
```

Next we open our persistent HDF5 file system and save transformed and preprocessed data there so we will not need to preprocess it every time we invoke algorithm. Also what is important is if we are just creating hdf5 file we use 
h5open(pathToHDF5, "w") and if it already exist fid = h5open(pathToHDF5, "r+") 


```
#!!!!!!!!!! important if you are just creating the hdf5 file  do it with "w" option otherwise do it with "r+"
fid = h5open(pathToHDF5, "w")
#fid = h5open(pathToHDF5, "r+") 
gr= getGroupOrCreate(fid, patienGroupName)

#for this particular example we are intrested only in liver so we will keep only this label
labelArr=map(entry-> UInt32(entry==1),loaded[2])


#we save loaded and trnsformed data into HDF5 to avoid doing preprocessing every time
saveMaskBeforeVisualization(fid,patienGroupName,loaded[1],"image", "CT" )
saveMaskBeforeVisualization(fid,patienGroupName,labelArr,"labelSet", "boolLabel" )

# here we did default transformations so voxel dimension is set to 1,1,1 in any other case one need to set spacing attribute manually to proper value
# spacing can be found in metadata dictionary that is third entry in loadByMonaiFromImageAndLabelPaths output
# here metadata = loaded[3]
writeGroupAttribute(fid,patienGroupName, "spacing", [1,1,1])

```
Now we prepare definitions that will be used to generate additional arrays that we will use as output of our algorithm and imput to manual modifications.

```
#just needed so we will not have 2 same colors for two diffrent informations
listOfColorUsed= falses(18)

##below we define additional arrays that are not present in original data but will be needed for annotations and storing algorithm output 

#manual Modification array
manualModif = MedEye3d.ForDisplayStructs.TextureSpec{UInt8}(# choosing number type manually to reduce memory usage
    name = "manualModif",
    color =getSomeColor(listOfColorUsed)# automatically choosing some contrasting color
    ,minAndMaxValue= UInt8.([0,1]) #important to keep the same number type as chosen at the bagining
    ,isEditable = true ) # we will be able to manually modify this array in a viewer

algoVisualization = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(
    name = "algoOutput",
    # we point out that we will supply multiple colors
    isContinuusMask=true,
    colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
    ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
   )

    addTextSpecs=Vector{MedEye3d.ForDisplayStructs.TextureSpec}(undef,2)
    addTextSpecs[1]=manualModif
    addTextSpecs[2]=algoVisualization

```

Time to display data

```
mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)

```
after some scrolling (you can press f for fast scrolling) - more details how to operate viewer in https://github.com/jakubMitura14/MedEye3d.jl

You should see sth like below

![image](https://user-images.githubusercontent.com/53857487/160683323-efc0efae-3742-4ec1-b330-dbdde6dc1075.png)


Now time for anotations - just drag the mouse pressing left mouse key over the liver you will get sth like below

![image](https://user-images.githubusercontent.com/53857487/160683751-27983862-c04b-4dee-bc3f-b0bf397b4c74.png)



Time for proper algorithm - we will use data from you annotations to construct ptobability distributions that will summarize local liver properties.

```

to be continued

```
















