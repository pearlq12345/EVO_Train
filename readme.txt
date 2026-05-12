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
  - 如果部署到公网，建议设置鉴权环境变量：
    export EVO_TRAIN_CLIENT_TOKEN='<roboclaw-client-token>'
    export EVO_TRAIN_ADMIN_TOKEN='<admin-token>'
    export EVO_TRAIN_ALLOW_RAW_COMMAND=false
    用户侧请求带 apiToken/token；管理员请求带 adminToken/apiToken/token。
    不设置这些环境变量时保持本地开发兼容，不强制鉴权。

  0. wallet / billing actions
     - action="余额查询"
       请求：{"username":"u","action":"余额查询","apiToken":"<roboclaw-client-token>"}
       返回 wallet / tasks
     - action="账单查询"
       请求：{"username":"u","action":"账单查询"}
       返回 wallet / billingRecords / tasks
     - action="管理员充值"
       请求：{"username":"u","action":"管理员充值","balanceCents":10000,"adminToken":"<admin-token>"}
       返回 wallet / billingRecords / tasks
     - action="价格设置"
       请求：{"action":"价格设置","provider":"aliyun","gpuSpec":"default","hourlyPriceCents":1000,"adminToken":"<admin-token>"}
       返回 provider / gpuSpec / hourlyPriceCents
     - action="价格查询"
       请求：{"action":"价格查询","provider":"aliyun"}
       返回 prices
     - action="平台余额查询"
       请求：{"action":"平台余额查询","provider":"autodl","minimumAssets":1000,"adminToken":"<admin-token>"}
       返回 AutoDL token 对应账户余额；assets / 1000 = 元，lowBalance=true 时应提醒管理员充值

  0.5. RoboClaw / AI planner actions
     - action="AI配置训练"
       用于 RoboClaw 这类 AI 入口：用户说自然语言，AI/后端生成可确认的 workflow plan，不创建实例、不扣费。
       请求：
       {"username":"u","action":"AI配置训练","message":"我想在metaworld上跑pick-place，20个epoch，训练后评估10个episode","provider":"autodl"}
       返回 plan：
       workflow / provider / params / command / workdir / checkpointPath / datasetPath / gpuSpec /
       hourlyPriceCents / estimatedHours / estimatedMinimumCostCents / missingFields / warnings /
       readyToStart / summary / needsConfirmation
     - 用户确认后，前端把 plan 的 workflow/params 交给 action="开始训练"。
     - 如果 missingFields 非空，RoboClaw 应继续追问用户，不应该开始训练。
     - warnings 用于给用户确认预算、GPU、超参风险；estimatedMinimumCostCents 是最低预冻结费用。

  1. action="开始训练"
     - 当前支持 provider=aliyun / provider=autodl
     - 请求里可带 datasetPath / checkpointPath / epochs / checkpointFrequency / gpuCount
       以及 image / ecsSpec / workspaceId 等阿里云参数
     - 生产推荐走 workflow/params；裸 command 只用于高级调试，需要设置 EVO_TRAIN_ALLOW_RAW_COMMAND=true
     - AutoDL 高级调试可直接传 command / workdir：
       {"username":"u","taskName":"eval-1","action":"开始训练","provider":"autodl","command":"python eval.py --suite libero_object_task","workdir":"/root/autodl-tmp/evf"}
     - AutoDL 托管实例模式可传：
       {"username":"u","taskName":"eval-1","action":"开始训练","provider":"autodl","command":"python eval.py","workdir":"/root/autodl-tmp/evf","autodlManaged":true,"autodlInstanceUuid":"pro-xxxx"}
     - 产品模式下，普通用户不需要传 AUTODL_HOST / AUTODL_PORT / AUTODL_KEY_PATH。
       后端会用团队 AUTODL_TOKEN 创建/开机实例，并通过 AutoDL snapshot 自动读取 proxy_host / ssh_port / root_password 执行训练。
     - 也支持 workflow/params，由后端 recipe 自动生成 command：
       {"username":"u","taskName":"mw-1","action":"开始训练","provider":"autodl","workflow":"evf_metaworld","params":{"envName":"pick-place-v2","epochs":20,"evalEpisodes":10}}
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

  5. action="结果下载"
     - 当前 AutoDL provider 支持 JSON 分块下载：服务端在远端把 artifactPath/checkpointPath 打成 tar.gz，
       每次返回一段 base64，客户端用 nextOffset 继续拉下一块，done=true 表示结束
     - 请求：
       {"username":"u","taskName":"eval-1","action":"结果下载","artifactPath":"/root/autodl-tmp/evo_train/output","offset":0,"chunkSize":1048576}
     - 返回 artifact：
       artifactPath / archivePath / offset / nextOffset / chunkSize / totalBytes / done / dataBase64
     - 这个接口保持 EVO-Train 的 JSON request + \n 协议，不需要另开 HTTP 文件服务

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

enterprise api / platform account：
  1. 推荐账户模型
     - 我们团队申请/持有 AutoDL 开发者或企业 API token，并给这个平台账户预充值
     - 用户不直接拿 AutoDL token；用户只在 RoboClaw/EVO-Train 充值到我们的 user_wallets
     - 用户开始训练时扣我们系统里的用户余额；底层实际消耗 AutoDL 平台账户余额
     - AUTODL_MIN_ASSETS 是平台账户安全线，低于阈值就拒绝新托管实例，避免平台账户被打穿

  2. 资金流
     - 用户支付成功 -> 支付回调/管理员后台写 user_wallets + billing_records
     - 开始训练 -> 冻结用户至少 1 小时费用
     - AutoDL 创建/开机/跑训练 -> 消耗我们团队 AutoDL 账户
     - 每小时续冻用户余额；用户余额不足 -> 自动 stop AutoDL job，并按配置关机/释放实例
     - 管理后台定期调用 action="平台余额查询"，lowBalance=true 时提醒团队给 AutoDL 企业账户充值

  3. 生产要补的支付闭环
     - 当前代码已有管理员充值接口，适合内测和人工入账
     - 真正上线时，把微信/支付宝/Stripe 等支付回调接到同一张 user_wallets / billing_records 即可
     - 管理员充值接口必须配 EVO_TRAIN_ADMIN_TOKEN，不能裸露在公网

workflow / recipe：
  1. 目标分层
     - RoboClaw 做用户对话和 AI planner：把自然语言转成 workflow + params
     - EVO_Train 做训练执行引擎：把 workflow + params 转成 command，然后走 AutoDL/Aliyun provider
     - 用户不需要知道云厂商 token、SSH 地址、端口、密钥，也不需要自己拼复杂 command

  2. 当前内置 recipe
     - evf_metaworld
       params: envName / epochs / batchSize / learningRate / seed / evalEpisodes / saveVideo
     - evf_libero
       params: suite / taskId / epochs / batchSize / learningRate / seed / evalEpisodes / saveVideo

  3. 推荐交互
     - 用户：我想在 metaworld pick-place 跑 20 个 epoch，训练完评估 SR
     - RoboClaw -> EVO_Train: action="AI配置训练"
     - EVO_Train 返回 plan、缺失字段、风险提示和最低预冻结费用
     - 用户确认
     - RoboClaw -> EVO_Train: action="开始训练", workflow="evf_metaworld", params={...}
     - EVO_Train 冻结用户余额，开 AutoDL 实例，执行 recipe command，后续同步状态/下载结果

autodl：
  1. SSH runner 环境变量（手动调试/复用已有实例时使用）
     export AUTODL_HOST='<ssh-host>'
     export AUTODL_PORT='<ssh-port>'
     export AUTODL_USER='root'
     export AUTODL_KEY_PATH='/path/to/private-key'
     # 如果没有配置 SSH key，也可用密码：
     export AUTODL_PASSWORD='<ssh-password>'
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
     # 托管模式会通过 /api/v1/dev/instance/pro/snapshot 自动拿 SSH 连接信息，
     # 所以正式产品用户侧不需要知道 AutoDL 实例地址、端口、密码或密钥。

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
