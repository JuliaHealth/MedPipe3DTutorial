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


#representing number that is the patient id in this dataset
patienGroupName="0"
fileNumber=1

#directory of folder with files in this directory all of the image files should be in subfolder volumes 0-49 and labels labels if one ill use lines below
data_dir = "D:\\dataSets\\CTORGmini\\"
#directory where we want to store our HDF5 that we will use
pathToHDF5="D:\\dataSets\\forMainHDF5\\smallLiverDataSet.hdf5"

train_labels = map(fileEntry-> joinpath(data_dir,"labels",fileEntry),readdir(joinpath(data_dir,"labels"); sort=true))
train_images = map(fileEntry-> joinpath(data_dir,"volumes 0-49",fileEntry),readdir(joinpath(data_dir,"volumes 0-49"); sort=true))


#zipping so we will have tuples with image and label names
zipped= collect(zip(train_images,train_labels))
tupl= zipped[fileNumber]
#proper loading
loaded = LoadFromMonai.loadByMonaiFromImageAndLabelPaths(tupl[1],tupl[2])
#now We open the hdf5 and save the array with some required metadata

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

#******************for display
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


#2) primary display of chosen image 
mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)
mainScrollDat.dataToScroll[1].name

ctArr=getArrByName("image" ,mainScrollDat)

subs = ctArr[1:50,1:50,1:50]

using Wavelets



xt = dwt(subs, wavelet(WT.db2))


aaa= rand(16,16)

wt = wavelet(WT.Coiflet{4}(), WT.Filter, WT.Periodic)
xdt = dwt(aaa, wt)
xdt


ss= dwt(aaa, wavelet(WT.db2))