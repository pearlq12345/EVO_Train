# Skill: Runtime Monitor

## Purpose

Monitor an active Training Task, keep the user informed, and protect budget.

## When To Use

Use this skill after `开始训练` succeeds.

## Backend Actions

Use:

- `查询状态`
- `任务同步`
- `账单查询`
- `请求用户日志`
- `结束训练`

## Monitor Loop

At each interval:

1. Query status.
2. Query billing records and wallet.
3. If status is failed, unknown, or stopped unexpectedly, collect logs.
4. If balance is near the next hourly freeze, warn the user.
5. If wallet policy requires stop, call `结束训练`.

## User-Facing Summary

Do not dump raw provider states unless needed. Translate into:

- running;
- waiting for instance;
- finished successfully;
- failed and needs diagnosis;
- stopped for billing or user request.

Example:

```text
任务仍在运行。首小时费用已冻结，当前余额足够继续下一小时。日志没有出现明确失败信号。
```

## Escalation Rules

- If `查询状态` fails twice, ask for provider metadata or retry later.
- If `请求用户日志` returns useful error text, switch to `failure-diagnosis`.
- If the task is running but artifacts never appear after expected runtime, flag as possible command or output-path issue.

## Done Criteria

- Current status is known.
- Billing state is known.
- Next action is clear: continue, diagnose, download, or stop.

