start：
      python3 server_tcp/server_connection.py --host 0.0.0.0 --port 9000 --workers 4
      # 后台计费扫描默认开启，每 300s 扫描一次 Running/未结算任务：
      # python3 server_tcp/server_connection.py --billing-scan-interval 60
      # 如需本地调试关闭：
      # python3 server_tcp/server_connection.py --disable-billing-scheduler
structure：
                              ┌────────────────────┐
                                      Client
                             roboclaw 训练中心-在线训练
                                JSON request + \n  
                              └─────────┬──────────┘
                                        │ TCP 长连接
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     server_connection.py                              
│               reactor-actor 模型响应用户链接/请求                                                                                     
│  1. 监听端口 9000                                                    
│  2. accept 客户端连接                                                
│  3. selector/epoll 管理 socket 事件，建立链接 / 响应任务              
│  4. 维护长连接 idle timeout                                           
│  5. 生成 TrainTaskEvent，交给线程池                                   
└───────────────────────────────┬──────────────────────────────────────┘
                                │ submit(event)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         thread_pool.py                              
│                  线程池，启动 4/8 个 worker 线程                                                                                   
│  1. 维护任务队列，线程消费 train_task_queue                          
│  2. worker 消费 TrainTaskEvent                                       
│  3. 调用业务函数 handle_request 处理请求                              
└───────────────────────────────┬──────────────────────────────────────┘
                                │ 数据库管理用户任务信息 sql_xxx()
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          sql_pack.py                                                                                                      
│  1. SQLite 初始化                                                    
│  2. sql_get_user_all_task    获取一个用户的所有任务                  
│  3. sql_add_user_task        为一个用户添加任务                       
│  4. sql_delete_user_task     删除一个用户的某个任务                   
│  5. 数据默认持久化到：sql_lite/data/tasks.sqlite3，可用 EVO_TRAIN_TASK_DB 覆盖
└─────────────────────────────────────────────────────────────────────┘

current request / response notes：
  0. wallet / billing actions
     - action="余额查询"
       请求：{"username":"u","action":"余额查询"}
       返回 wallet / tasks
     - action="账单查询"
       请求：{"username":"u","action":"账单查询"}
       返回 wallet / billingRecords / tasks
     - action="管理员充值"
       请求：{"username":"u","action":"管理员充值","balanceCents":10000}
       返回 wallet / billingRecords / tasks
     - action="价格设置"
       请求：{"action":"价格设置","provider":"aliyun","gpuSpec":"default","hourlyPriceCents":1000}
       返回 provider / gpuSpec / hourlyPriceCents
     - action="价格查询"
       请求：{"action":"价格查询","provider":"aliyun"}
       返回 prices
     - action="平台余额查询"
       请求：{"action":"平台余额查询","provider":"autodl","minimumAssets":1000}
       返回 AutoDL token 对应账户余额；assets / 1000 = 元，lowBalance=true 时应提醒管理员充值

  1. action="开始训练"
     - 当前支持 provider=aliyun / provider=autodl
     - 请求里可带 datasetPath / checkpointPath / epochs / checkpointFrequency / gpuCount
       以及 image / ecsSpec / workspaceId 等阿里云参数
     - AutoDL 可直接传 command / workdir：
       {"username":"u","taskName":"eval-1","action":"开始训练","provider":"autodl","command":"python eval.py --suite libero_object_task","workdir":"/root/autodl-tmp/evf"}
     - AutoDL 托管实例模式可传：
       {"username":"u","taskName":"eval-1","action":"开始训练","provider":"autodl","command":"python eval.py","workdir":"/root/autodl-tmp/evf","autodlManaged":true,"autodlInstanceUuid":"pro-xxxx"}
     - 服务端会提交真实 provider 任务，并把 provider / jobId / checkpointPath / datasetPath
       一起写入 SQLite
     - 训练类接口返回会同时带 wallet，供前端刷新余额

  2. action="任务同步"
     - 会返回当前用户的任务列表
     - 会按 provider 刷新远端 status，并回写到 SQLite

  3. action="结束训练"
     - 会按 provider stop 远端 job，再把本地任务状态标成 STOPPED

  4. action="删除任务"
     - 只删除本地 SQLite 记录，不会再额外 stop 远端任务

billing：
  1. 当前是最小预付费模型，还没有接真实支付
     - 用户开始训练前，钱包可用余额必须 >= 1 小时预估费用
     - 开始训练时先冻结 1 小时费用
     - 任务同步发现任务结束时，会结算已冻结费用
     - 后台 billing scheduler 会定期扫描未结算任务
     - 任务运行超过 frozenUntil 时，会尝试继续冻结下一小时；余额不足会按 provider 自动 stop 远端任务并标记 STOPPED
     - AutoDL 与 Aliyun 共用同一套 wallet / gpu_prices / billing_records
     - 平台自身 AutoDL 账户余额也可监控：AUTODL_MIN_ASSETS 低于阈值时拒绝创建托管实例，并可通过 action="平台余额查询" 给管理后台展示

  2. 默认小时价格
     - 可通过环境变量设置默认价格，单位是 cents：
       export EVO_TRAIN_DEFAULT_HOURLY_PRICE_CENTS=1000
     - 也可以在请求里传 hourlyPriceCents 覆盖本次任务价格

  3. 充值/价格配置
     - 当前提供的是后端函数 sql_set_user_balance / sql_set_gpu_price，供管理脚本或测试调用
     - TCP action="管理员充值" / action="价格设置" 可用于第一版管理后台
     - 后续真实充值系统只需要写 user_wallets 和 billing_records 即可

autodl：
  1. SSH runner 环境变量
     export AUTODL_HOST='<ssh-host>'
     export AUTODL_PORT='<ssh-port>'
     export AUTODL_USER='root'
     export AUTODL_KEY_PATH='/path/to/private-key'
     export AUTODL_WORKDIR='/root/autodl-tmp/evf'

  2. AutoDL API 托管实例环境变量
     export AUTODL_TOKEN='<developer-token>'
     export AUTODL_INSTANCE_UUID='pro-xxxx'          # 可选，复用已有实例
     export AUTODL_GPU_SPEC_UUID='pro6000-p'         # 创建新实例时需要
     export AUTODL_IMAGE_UUID='image-xxxx'           # 创建新实例时需要
     export AUTODL_DATA_CENTER_LIST='westDC3,beijingDC2'
     export AUTODL_MIN_ASSETS=1000                   # assets/1000=元，低于阈值拒绝新任务
     export AUTODL_POWER_OFF_ON_STOP=true            # 停任务后自动关机
     export AUTODL_RELEASE_ON_STOP=false             # 停任务后是否释放实例

task record fields：
  - taskName
  - status
  - provider
  - jobId
  - checkpointPath
  - datasetPath
  - hourlyPriceCents
  - frozenUntil
  - startedAt
  - stoppedAt
  - actualCostCents
  - billingStatus
  - error
  - createdAt
  - updatedAt
