## Template for processing ACE Explorer data with aceR
## See the analysis resources on https://sites.google.com/view/ace-explorer-researchers/ for more details and tutorials on aceR
## Last updated 2022-11-17 by Kristine O'Laughlin
## Last updated 2021-04-23 by Jessica Younger

## GUIDE:
## Login to https://ucsf-ace-prod.firebaseapp.com/
## Select sessions from Users with Prefix: dys
## Click on all the reports on the right to download them all
  ## (18 reports total)
## Put all reports into a single directory on your computer
  ## (i.e. "path_in" variable below)
## NOTE: Make sure only the current, updated CSVs are present
  ## (i.e. no older reports in the same directory!)
## Specify the output directory (i.e. "path_out" variable below)
## Then, run this script.
## At the end, you should have two CSV files in your output directory:
  ## 1) Formatted in ACE Explorer format (i.e. "ace_averaged_data_DATE.csv")
  ## 2) Formatted in REDCap format (i.e. "ace_averaged_data_DATE_redcap.csv")

## NOTE: If this is your first time running this script,
  ## you will need to install the required packages below.
  ## Uncomment the "install" lines below and run them first.
  ## Then, comment them out again and run the rest of the script.

# INSTALL PACKAGES (FOR FIRST TIME USE):
#install.packages("ggplot2")
#install.packages("tidyverse")
#install.packages("devtools")
#install aceR package:
#devtools::install_github("joaquinanguera/aceR")

# LOAD PACKAGES:
library(ggplot2)
library(tidyverse)
library(aceR)
library(dplyr)

# BULK PROCESSING:
# Quick and Easy Analysis: use the summary function to process data according to our suggested best practices

#Specify the file path to the folder with your ace data
path_in=("L:/language/rbogley/ACE_Explorer/Bulk_Raw")
path_out=("L:/language/rbogley/ACE_Explorer/Bulk_Processed")

#########THIS CODE CORRECTS THE SPATIAL SPAN PROCSESING ERROR - Kristine O'Laughlin###################
# Set working directory to raw ace data reports
setwd(path_in)

# Read in ACE Spatial Span files
temp = list.files(pattern = 'Spatial Span', recursive = T)
myfiles = lapply(temp, read.csv)
names(myfiles) = temp

# Loop over Spatial Span files and shift columns for participants with missing Response.Window
for (file in names(myfiles)) {
  for (i in 1:nrow(myfiles[[file]])) {
    if (is.na(myfiles[[file]]$Total.Taps[i])) { 
      for (col in ncol(myfiles[[file]]):which(colnames(myfiles[[file]]) == 'Response.Window')) {
        myfiles[[file]][i, col] = myfiles[[file]][i, col - 1]}
      if (myfiles[[file]]$Session.Type[i] == 'StartingWindow') {
        myfiles[[file]]$Response.Window[i] = -1}
      else {
        myfiles[[file]]$Response.Window[i] = 8000 + (myfiles[[file]]$Object.Count[i] - 3)*1000}}}}

# CHECK THAT THE MYFILES OBJECT IS NOT EMPTY
summary(myfiles) # THIS SHOULD BE A LIST INCLUDING 2 DATA.FRAMES
# CHECK THAT THE ENVIRONMENT TAB SHOWS "Large list (2 elements, x.x MB)" UNDER "Data" HEADING

# DO NOT RUN IF myfiles OBJECT IS EMPTY
# Export Spatial Span files with corrected column alignment
for (i in names(myfiles)) {
  write.csv(myfiles[[i]], i, row.names = F)}
##########FIXING TNT
# Set working directory to raw ace data reports
setwd(path_in)

# Read in ACE TNT files
temp = list.files(pattern = 'TNT', recursive = T)
myfiles = lapply(temp, read.csv)
names(myfiles) = temp

# Remove last 4 columns that cause error in TNT
for (file in names(myfiles)) {
  if (ncol(myfiles[[file]]==38)) {
    myfiles[[file]]=myfiles[[file]][,1:34]}}

# Overwrite TNT csv without last 4 columns
for (i in names(myfiles)) {
  write.csv(myfiles[[i]], i, row.names = FALSE)}

##############

#Use the summary wrapper function to process the data and get summary metrics for each module 
#Use ?proc_ace_complete to see additional options 
data=proc_ace_complete(path_in=path_in, data_type="explorer", path_out=path_out)
#a csv file will be saved in the parent directory of the path if no output directory is specified. That's it!

#############Exclude Test Cases (containing "test" in PID/BID)
data=data[-grep('test',data$pid,ignore.case=TRUE),]
# Exclude Test Case that is called "dys12345"
data=data[-grep('dys12345',data$pid,ignore.case=TRUE),]
# Exclude any Test Case that has a pid of "dys000":
data=data[-grep('dys000',data$pid,ignore.case=TRUE),]

# write.csv(data, file = paste0(path_out,"/ace_averaged_data_",Sys.Date(),".csv"),row.names=FALSE)

################################################################################################################
# Copy the data to from data to a dataframe called "data_redcap":
data_redcap = data

# Remove unnecessary columns:
data_redcap = data_redcap %>% 
  select(-c("age",
            "handedness",
            "BRT.rt_mean.correct.dominant",
            "BRT.rt_mean.correct.dominant.thumb",
            "BRT.rt_mean.correct.nondominant",
            "BRT.rt_mean.correct.nondominant.thumb"))
colnames(data_redcap)

# Create a data dictionary to change column names, from the ACE names to REDCap names:
data_dict = data.frame(
  ace_name = c("pid",
               "bid",
               "BOXED.rcs.overall",
               "COLORSELECTION.max_delay_time.correct.strict",
               "FLANKER.rcs.overall",
               "SAATSUSTAINED.rt_mean.correct",
               "SAATIMPULSIVE.rt_mean.correct",
               "TNT.rt_mean.correct",
               "BRT.rt_mean.correct",
               "SPATIALSPAN.object_count_span.overall",
               "BACKWARDSSPATIALSPAN.object_count_span.overall",
               "SPATIALCUEING.rcs.overall",
               "FILTER.k.r2b0",
               "FILTER.k.r2b2",
               "FILTER.k.r2b4",
               "FILTER.k.r4b0",
               "FILTER.k.r4b2"),
  redcap_name = c("id_number",
                  "redcap_event_name",
                  "ace_boxed",
                  "ace_colorswatch",
                  "ace_flanker",
                  "ace_sustained",
                  "ace_impulsive",
                  "ace_tnt",
                  "ace_brt",
                  "ace_spatialspan_f",
                  "ace_spatialspan_b",
                  "ace_compass",
                  "ace_filter_2b0",
                  "ace_filter_2b2",
                  "ace_filter_2b4",
                  "ace_filter_4b0",
                  "ace_filter_4b2")
)

# Replace column names with their corresponding new names:
data_redcap = data_redcap %>% 
  rename_at(vars(data_dict$ace_name), ~data_dict$redcap_name)

colnames(data_redcap)


# In the column "redcap_event_name", remove the information before the string "session" for each value with blank:
data_redcap$redcap_event_name = gsub(".*session", "", data_redcap$redcap_event_name)

# Show all the unique values in the column "redcap_event_name":
unique(data_redcap$redcap_event_name)

# Replace the values in the column "redcap_event_name" with the following values:
redcap_event_dict = data.frame(
  ace_event = c(0,
                1,
                2,
                3,
                4),
  redcap_event = c("visit_one_arm_1",
                   "visit_two_arm_1",
                   "visit_three_arm_1",
                   "visit_four_arm_1",
                   "visit_five_arm_1")
)

# Replace the values in the column "redcap_event_name" with the corresponding values in the dictionary using gsub:
data_redcap$redcap_event_name = gsub(paste0(redcap_event_dict$ace_event, collapse = "|"), redcap_event_dict$redcap_event, data_redcap$redcap_event_name)

# Remove the "dys" prefix to every value in the "id_number" column:
data_redcap$id_number = gsub("dys", "", data_redcap$id_number)

# Print the dataframe:
data_redcap

unique(data_redcap$redcap_event_name)

data_redcap

# Save the data to the csv file but append "_redcap" to the file name:
write.csv(data_redcap, file = paste0(path_out,"/ace_averaged_data_",Sys.Date(),"_redcap.csv"),row.names=FALSE)

# Read the CSV file:
csv_data = readLines(paste0(path_out,"/ace_averaged_data_",Sys.Date(),"_redcap.csv"))

# Replace NA with blanks:
csv_data <- gsub("NA", "", csv_data)

# Write the modified CSV data back to the file
writeLines(csv_data, paste0(path_out,"/ace_averaged_data_",Sys.Date(),"_redcap.csv"))

# IMPORTANT NOTE: PLEASE CHECK YOUR CSF PRIOR TO UPLOADING
# CHECK THAT THE DATA/COLUMN NAMES HAVE BEEN SUCCESFULLY MODIFIED TO REDCAP VARIABLES
# CHECK FOR DUPLICATES AND CORRECT THEM
# REPLACE NA's WITH BLANKS IN DATA IF NOT ALREADY DONE
# REMOVE ANY TEST/NOT REAL CASES THAT MAY HAVE SLIPPED THROUGH (THEN UPDATE ABOVE TO REMOVE THEM MOVING FORWARD)
# FINALLY, WHEN IMPORTING TO REDCAP - CHECK FOR ANY DIFFERENCES IN THE DATA AND MAKE SURE THEY ARE CORRECT
# THANKS! :)

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################



# ###########Step by Step Processing: walk through each step to customize the processing steps
# ##Proc_ace_complete calls each of the below functions 

# #enter PIDN below after creating folder with same name in ACE_Explorer/Participants filepath and downloading all testing CSV's
# PIDN=("34249")

# ####Read in Raw Data from ACE output
# #path to folder with raw data output
# path=paste0("L:/language/rbogley/ACE_Explorer/New_Participants/",PIDN)
# #path=paste0("L:/language/rbogley/ACE_Explorer/Participants")

# #path to export individual report summaries
# end_path=("L:/language/rbogley/ACE_Explorer/Case_Processed")
# #bulk load the data from all CSV's
# data=load_ace_bulk(path, data_type = 'explorer')

# #OPTIONAL data cleaning: Use the trim_rt_trials functions to specify min and max RTs that should be included and/or a value for 
# #standard deviation to remove trials outside of the specified number of standard deviations outside of an individual's mean RT
# #Trials with an RT < 150ms are already excluded, and Backward and Forward Spatial Span (Gem Chaser) will be excluded from RT cleaning (since RT is not relevant for this task)
# #To exclude additional modules from cleaning, simply enter the module name in quotes in the 'exclude' argument.
# #We recommend excluding SAAT data from outlier RT cleaning, as these outliers are a key factor
# #in assessing SAAT performance. To perform cleaning for this module, remove "SAAT" from the 'exclude' argument
# data=trim_rt_trials_range(data, cutoff_min =200, cutoff_max=NA, exclude=c())
# data=trim_rt_trials_sd(data, cutoff=3, exclude=c("SAAT"))

# #Feed the data variable from above directly in the proc_by_module script
# #do processing on data, assign to a variable. See proc_by_module documentation for more options
# data_averages = proc_by_module(data, verbose=TRUE, app_type = 'explorer')

# ####Data Cleaning####
# ###If skipping any optional steps, please make sure you change the variable names appropriately.
# #For example, if you skip line 50, line 55 will need 'data_averages' in place of 'data_averages_scrub'  

# # data_averages will be one data frame with data from ALL processed modules. Check the structure of the output 
# str(data_averages)

# #OPTIONAL data cleaning: Use post_clean_low_trials to replace data that has fewer than the specified number of trials per condition with NA
# data_averages_scrub = post_clean_low_trials(data_averages, app_type = "explorer",extra_demos = c(), min_trials = 5)

# #OPTIONAL data cleaning: Use post_clean_chance to replace data with NA if subject performed AT or BELOW a given cut off level on a given module
# #Please enter the minimum acceptable accuracy for each type of response, dprime, two-option forced choice, and four-option forced choice
# #also specify if you want criteria to be evaluated based on overall accuracy of the module (overall = TRUE)  or the easy condition only (overall = FALSE)
# data_averages_rmchance = post_clean_chance(data_averages_scrub, app_type="explorer", extra_demos = c(), overall = TRUE, cutoff_dprime = 0, cutoff_2choice = 0.5, cutoff_4choice = 0.25)

# #OPTIONAL: Use post_reduce_cols to select only the metrics of interest. You can add or subtract to these names.
# #Enter demographic related data that are constant across modules and that should be preserved in the reduced data frame
# #You can use names(data_averages) to get a sense of how columns are named. This script will select the columns that match the strings entered into the function

# #The code here will output only the suggested metrics of interest for each module. You can edit the metrics_names options to be more general to get additional info for each module. For example, use "rt_mean.correct" to get mean rt for all modules 
# data_averages_reduced = post_reduce_cols(data_averages_rmchance,  demo_names = c("pid", "age", "handedness", "bid"), metric_names =  c("BRT.rt_mean.correct","SAAT.rt_mean.correct", "SAATIMPULSIVE.rt_mean.correct", "SAATSUSTAINED.rt_mean.correct", "TNT.rt_mean.correct", "object_count_span.overall", "FILTER.k", "rcs.overall"))

# #You might want to reduce the variables even further, because things like RT for spatial span are not really relevant. 
# data_averages_reduced = data_averages_reduced[, !grepl("TNT.rt_mean.correct.", names(data_averages_reduced))]
# data_averages_reduced = data_averages_reduced[, !grepl("FILTER.rcs", names(data_averages_reduced))]

# #Write out your processed data to the output path specified above
# write.csv(data_averages_reduced, file=paste(end_path,"/", PIDN, "_ACE_averaged_data", ".csv", sep=""),row.names=FALSE)




