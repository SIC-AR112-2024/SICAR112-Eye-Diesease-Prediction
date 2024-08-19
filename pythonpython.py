import os
 
# Function to rename multiple files
def main():
   
    folder = r"C:\Users\swaagat\Downloads\SICAR112-Eye-Diesease-Prediction\dataset\diabetic_retinopathy"
    i=0
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"{count}.jpg"
        src =f"{folder}\{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}\{dst}"
        
        # rename() function will
        # rename all the files
        # print(dst)
        os.rename(src, dst)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()