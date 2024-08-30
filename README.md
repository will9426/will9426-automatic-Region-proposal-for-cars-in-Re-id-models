# automatic-Region-proposal-for-cars-in-Re-id-models

## ResNet-Based Region Proposal Approach

Our approach proposes the use of a reduced region proposal module based on the same ResNet-based backbone. This method suggests a simpler approach using branches for both local and global attributes.

![model](images/model.png)


## Environment Setup

To set up the environment, run the following command to install all necessary dependencies:

```bash
pip install -r requirements.txt
```
### Dataset

### Train

#### Pesos


| Models      | Description                            | Link                                                                                                     |CMC K=5|CMC K=5 change context|
|-------------|----------------------------------------|----------------------------------------------------------------------------------------------------------|-------|----------------------|
| PGAN	      | All vric                               | [Link to Model A](https://drive.google.com/file/d/1ZSJwGtm0avQab9Tb1QSYFnQHVVRjPU3d/view?usp=drive_link) |93.1   |56.9                  |
| PGAN	      | Small vric                             | [Link to Model A](https://drive.google.com/file/d/1A2CsEjNyMPdZSBVXsgCoxSEkDu99boz9/view?usp=drive_link) |60.1   |                      |
| PGAN	      | All veRi                               | [Link to Model A](https://drive.google.com/file/d/1XWMifTM4l1jNozStG9E42IfstWr4nqYi/view?usp=drive_link) |97.5   |                      |   
| Ours        | All vric                               | [Link to Model B](https://drive.google.com/file/d/1z60rveZ6hOt0-8ISFajIw75DWkObHx9-/view?usp=drive_link) |89.8   |64.4                  |
| Ours        | Small vric                             | [Link to Model B](https://drive.google.com/file/d/1-BHt1-5Xxq3_XgU31jWiT_GOjWHt1t9j/view?usp=drive_link) |74.9   |53.3                  |
| Ours        | All veRi                               | [Link to Model B](https://drive.google.com/file/d/1xje1VY5VDAo46VTCn0NhN81xsHWV13Hu/view?usp=drive_link) |97.2   |                      |

#### Test

##### Run evaluator


##### Generate GradCAM



#### Generate qualitative Result

![model](images/inference.png)
