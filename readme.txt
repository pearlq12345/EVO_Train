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
│  1. 阿里云 RDS MySQL 初始化                                          
│  2. sql_get_user_all_task    获取一个用户的所有任务                  
│  3. sql_add_user_task        为一个用户添加任务                       
│  4. sql_delete_user_task     删除一个用户的某个任务                   
│  5. 通过 EVO_TRAIN_DATABASE_URL 或 DATABASE_URL 连接数据库            
└─────────────────────────────────────────────────────────────────────┘

database：
      pip install PyMySQL
      export EVO_TRAIN_DATABASE_URL='mysql+pymysql://user:password@rm-xxxx.mysql.rds.aliyuncs.com:3306/evo_train?charset=utf8mb4'

      # 兼容 evo-data_backend 的配置命名，也可以使用：
      export DATABASE_URL='mysql+pymysql://user:password@rm-xxxx.mysql.rds.aliyuncs.com:3306/evo_train?charset=utf8mb4'
