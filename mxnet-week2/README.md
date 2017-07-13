# MXNET-Week2

## Useful git commands
 
    git init # init a repo 
    git status # check the status of current branch 
    git clone # clone a repo from github 
    git add # add untracked files 
    git commit # make the commit change to the branch 
    git push # push the changes to github 
    git pull # pull the changes from github 

## Pycharm feature for gtihub 
    Version Control 
    Mark down for readme 
    Terminal 
    
## Setup AWS account 

### AWS website 
    https://aws.amazon.com/
### Sign Up

    Email account
    Credit Card 
    ~50 us dollar budget for our project if GPU is in your plan

### AWS Deep Learning pre-built instance 
    https://aws.amazon.com/marketplace/pp/B01M0AXXQB
    
### Steps to start a GPU instance in AWS 
    1) Sign up a AWS account with credit card bounded 
    2) Search for the Deep Learning AMI (free) by the link above
    3) Start a new instance from Deep Learnining AMI 
        1) US West (Oregon)	ami-296e7850 -> Lanuch with
        2) Enable public ip address
        3) select one of them 
            g2.2xlarge g2.8xlarge p2.xlarge p2.2xlarge p2.8large p2.16xlarge
        4) Security Group -> default 22, need 8888 6006 in future 
        5) For storage, 100G is necssaray 
        6) Create and download key-pair, such as "*.pem"
        7) run "ssh -i ~*.epm ec2-user@publicip"
        8) when in the instance, run "nvidia-smi" and "nvcc --version" to check the availbility of GPU
        
### IMPORTANT NOTICE 
    !!! STOP THE INSTSNCE ONCE YOU DON'T NEED IT
    !!! TERMINATE THE INSTANCE WHEN YOU FINSHED THE PROJECT 
    !!! UNLINK THE CREDIT CARD INFO FROM AWS ONCE YOU DON'T USE AWS
    !!! IT COST MONEY FOR EVERY SECOND YOU USE THE INSTANCES

### Useful info

Change the type of an instance 
    Right click on the instance -> Instance Setting -> Change Instance Type 
     