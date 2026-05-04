start：
      python3 server_tcp/server_connection.py --host 0.0.0.0 --port 9000 --workers 4
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
│  5. 数据持久化到：/Users/hongru/project/EVO_Train/sql_lite_data       
└─────────────────────────────────────────────────────────────────────┘

current request / response notes：
  1. action="开始训练"
     - 当前先支持 provider=aliyun
     - 请求里可带 datasetPath / checkpointPath / epochs / checkpointFrequency / gpuCount
       以及 image / ecsSpec / workspaceId 等阿里云参数
     - 服务端会提交真实 DLC 任务，并把 provider / jobId / checkpointPath / datasetPath
       一起写入 SQLite

  2. action="任务同步"
     - 会返回当前用户的任务列表
     - 对 aliyun 任务会顺手刷新一次远端 status，并回写到 SQLite

  3. action="结束训练"
     - 对 aliyun 任务会先 stop 远端 job，再把本地任务状态标成 STOPPED

  4. action="删除任务"
     - 只删除本地 SQLite 记录，不会再额外 stop 远端任务

task record fields：
  - taskName
  - status
  - provider
  - jobId
  - checkpointPath
  - datasetPath
  - error
  - createdAt
  - updatedAt
