# Backup e restauração PostgreSQL (Oráculo)

Antes de **formatar o servidor PostgreSQL** ou mudar `ORACULO_PG_HOST` no `.env`, faça um backup completo da base `oraculo` (ou o nome em `ORACULO_PG_DATABASE`).

## O que fica guardado

| Tabela | Conteúdo |
|--------|----------|
| `users` | Contas, passwords (hash), display_name, admin |
| `user_settings` | System prompt por utilizador, temperatura, etc. |
| `app_global` | Prompt global, llama-server, Runpod, fila de inferência |
| `chat_sessions` | Títulos e totais das conversas |
| `chat_messages` | Histórico de mensagens user/assistant |

## Pré-requisito

Cliente PostgreSQL com `pg_dump` / `pg_restore`:

```bash
sudo apt update && sudo apt install -y postgresql-client
```

Os scripts **não precisam** do venv do Oráculo nem de `psycopg2` — só leem `server_for_serveless/.env` e chamam `pg_dump`/`pg_restore`.

## 1. Fazer backup (no servidor ou num PC com acesso ao PG)

Na raiz do repositório, com `server_for_serveless/.env` configurado:

```bash
python3 server_for_serveless/scripts/pg_backup.py
```

Saída padrão: `server_for_serveless/backups/oraculo_YYYYMMDD_HHMMSS.dump`

Ficheiro custom (`pg_dump -Fc`), comprimido e adequado para `pg_restore`.

### Copiar o backup para fora do servidor

```bash
scp root@SEU_SERVIDOR:~/treinamento_llms/server_for_serveless/backups/oraculo_*.dump ./
```

Guarde **pelo menos duas cópias** (PC local + cloud), antes de formatar o disco ou reinstalar o PostgreSQL.

### Backup manual (sem script)

```bash
set -a && source server_for_serveless/.env && set +a
export PGPASSWORD="$ORACULO_PG_PASSWORD"
pg_dump -h "$ORACULO_PG_HOST" -p "${ORACULO_PG_PORT:-5432}" -U "$ORACULO_PG_USER" \
  -d "${ORACULO_PG_DATABASE:-oraculo}" --format=custom --no-owner --no-acl \
  -f oraculo_backup.dump
```

## 2. Depois de levantar o PostgreSQL de novo

1. Atualize `server_for_serveless/.env` com o novo `ORACULO_PG_HOST` (e password se mudou).
2. Confirme ligação:

   ```bash
   python3 -c "from server_for_serveless.db.pg_db import get_connection; get_connection().close(); print('OK')"
   ```

3. Restaure:

   ```bash
   python3 server_for_serveless/scripts/pg_restore.py --file server_for_serveless/backups/oraculo_YYYYMMDD_HHMMSS.dump
   ```

4. Suba o Oráculo:

   ```bash
   ./server_for_serveless/serve.sh
   ```

## 3. Fluxo recomendado antes de formatar

1. Parar o Oráculo (`pkill -f serve_lora` ou `systemctl stop …`).
2. `python3 server_for_serveless/scripts/pg_backup.py`
3. `scp` do `.dump` para o teu PC.
4. Só então formatar / reinstalar PostgreSQL.
5. Criar utilizador PG e base `oraculo` (ou deixar `ORACULO_PG_AUTO_CREATE_DATABASE=1`).
6. Restaurar com `pg_restore.py`.
7. Verificar login e uma sessão de chat antiga no admin.

## Notas

- A pasta `server_for_serveless/backups/` está no `.gitignore` — **não commite** dumps nem passwords.
- Backups `.dump` contêm **hashes de password** e **conversas** — trate como dados sensíveis.
- Para bases muito grandes, o mesmo comando funciona; o ficheiro cresce com o histórico de chats.
