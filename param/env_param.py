DEFAULT_REGION = "cn-hangzhou"
DEFAULT_WORKSPACE_ID = "604723"
DEFAULT_RESOURCE_ID = "quota8xa9l64xn9c"
DEFAULT_JOB_NAME = "job_for_mirror"
DEFAULT_JOB_TYPE = "PyTorchJob"
DEFAULT_ROLE = "Worker"
DEFAULT_IMAGE = "evo-train-mirror-registry-vpc.cn-hangzhou.cr.aliyuncs.com/evo-mirror-namespace/evo-mirror:v1-losslog"
DEFAULT_CPU = "10"
DEFAULT_GPU = "1"
DEFAULT_MEMORY = "50Gi"
DEFAULT_SHARED_MEMORY = "50Gi"
DEFAULT_WORKERS = 1
DEFAULT_VPC_ID = "vpc-bp1txl55oqch56uhbpc43"
DEFAULT_SWITCH_ID = "vsw-bp14hh8tnognnao2f79n2"
DEFAULT_SECURITY_GROUP_ID = "sg-bp171lhsibv9dwe0uia7"
DEFAULT_EXTENDED_CIDRS = ["172.19.0.0/16"]
DEFAULT_PRIORITY = 9
DEFAULT_ROUTE = "eth1"

OSS_CODE_DATA_SOURCE_ID = "d-hzpwiw5qvtyy7887oe" # 存放代码的OSS数据集ID
PAI_MOUNT_PATH_OF_CODE = "/mnt/code/" # 代码OSS数据集会被mount在PAI容器的哪个位置
# 由于容器无法使用clone，所以暂时将代码也用数据集的方式进行管理，直接挂载到pai的/mnt/code路径下

NAS_MODEL_RESULT_URI = "nas://001vtgf4opoobb8u5gh-vfp55.cn-hangzhou.nas.aliyuncs.com/" # 存放用户训练结果
PAI_MOUNT_PATH_OF_MODEL_RESULT = "/usrresult/" # pai容器下，用户训练结果存放的位置。也是nas挂载的位置。格式 /usrresult/usrname/taskname/
ECS_MOUNT_PATH_OF_MODEL_RESULT = "/mnt/usrresult" # NAS挂载在ECS的哪个路径下。
# pai 的/usrresult/ 和 ecs 的/mnt/usrresult 挂载的是同一个nas，在pai的/usrresult/usrname/taskname/下生成文件，
# 对应就在 ecs 的"/mnt/usrresult/%s/%s/checkpoint下生成文件
# 当时考虑用nas，是因为nas才支持像真正文件系统一样来读写，而oss虽然也能实现读写，但是无法进行链接操作（lerobot依赖链接操作生成训练结果。）
OSS_USR_DATASET_URI = "oss://evo-data.oss-cn-hangzhou-internal.aliyuncs.com/all_usr_dataset/%s/%s" 
PAI_MOUNT_PATH_OF_USR_DATASET = "/mnt/pai/data/" # 用户指定（或者默认）数据集在pai容器下的挂载路径
ECS_MOUNT_PATH_OF_USR_DATASET = "/home/evomind/usrdata/" # 用户的数据集挂载在ECS的哪个路径下，具体是格式 /home/evomind/usrdata/usrname/datasetname/
DEFAULT_USR_DATASET_ID = "d-xfobl8zdj3cqdrleqo" # 如果用户没有指定数据集，则使用默认数据集进行训练
# evo-data这个bucket下面的/all_usr_dataset/usrname/datasetname 是用户自己的数据集，所以在拉起pai训练任务的时候，需要将桶的某个路径同时挂载到pai 和 ecs