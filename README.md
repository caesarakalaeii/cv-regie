# cv-regie
Application based on open-cv and OBS to assist in switching between scenes and framing actors

## Goals:  
- Track and enumerate persons in multiple video feeds
- Find best suited feed to shoe person (size in frame, face visible?)
- crop and export data to OBS (crop in obs or cv? see #Potential Trouble makers)
- switch to multiple camera feeds when neccesary (Multiple ppl, suspended tracking see #Optionals)

## Optionals:  
- Use gesture control to suspend tracking of a certain feed



## Potential Trouble makers:  

- Webcams on Windows are exclusive!
    - Find workaround or export video feed directly from CV
