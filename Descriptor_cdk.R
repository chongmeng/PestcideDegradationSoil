library(rcdk)
library(openxlsx)
library(readxl)
library(tidyr)
library(dplyr)
# Read input data
d <- read_excel("your_file.xlsx")

# Define a list of SMILES strings for multiple molecules
smiles_list <- unique(d$SMILE)
mols<-parse.smiles(smiles_list)
mols <- mols[!sapply(mols, is.null)]
row_names<-names(mols)
allDescs <- data.frame(rowname=row_names)
dn<-c("org.openscience.cdk.qsar.descriptors.molecular.XLogPDescriptor",
  "org.openscience.cdk.qsar.descriptors.molecular.KappaShapeIndicesDescriptor",
  "org.openscience.cdk.qsar.descriptors.molecular.CarbonTypesDescriptor",
  "org.openscience.cdk.qsar.descriptors.molecular.HybridizationRatioDescriptor",
  "org.openscience.cdk.qsar.descriptors.molecular.HBondDonorCountDescriptor",             
  "org.openscience.cdk.qsar.descriptors.molecular.HBondAcceptorCountDescriptor",
  "org.openscience.cdk.qsar.descriptors.molecular.BondCountDescriptor",
  "org.openscience.cdk.qsar.descriptors.molecular.AtomCountDescriptor",                 
  "org.openscience.cdk.qsar.descriptors.molecular.AromaticBondsCountDescriptor",        
  "org.openscience.cdk.qsar.descriptors.molecular.AromaticAtomsCountDescriptor",
  "org.openscience.cdk.qsar.descriptors.molecular.APolDescriptor",
  "org.openscience.cdk.qsar.descriptors.molecular.ALOGPDescriptor",
  "org.openscience.cdk.qsar.descriptors.molecular.FractionalPSADescriptor"
  )

for (i in dn) {
  #dn<-get.desc.names(dc[i])
  Descs <- eval.desc(mols, i,verbose = TRUE)
  Descs<-mutate(Descs,rowname=rownames(Descs))
  allDescs <- merge(allDescs, Descs, by = "rowname")
}

allDescs <- allDescs[, colSums(is.na(allDescs)) != nrow(allDescs)]