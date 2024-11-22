import json
import cx_Oracle  # 오라클 데이터베이스 라이브러리

# 오라클 데이터베이스 연결 설정
dsn_tns = cx_Oracle.makedsn('localhost', 1521, service_name='db_test')
conn = cx_Oracle.connect(user='system', password='123456', dsn='xe')
cursor = conn.cursor()

# 쿼리 실행
cursor.execute("SELECT tags, patterns, responses FROM intents")
data = cursor.fetchall()

intents_dict = {}

# 데이터 전처리 및 JSON 형식으로 변환
for row in data:
    tag = row[0]
    patterns = row[1].split('|')  # 패턴이 여러 개일 경우 '|'로 구분
    response = [row[2]] 

    if tag not in intents_dict:
        intents_dict[tag] = {
            'tag': tag,
            'patterns': [],  # 초기화
            'responses': response  # 응답은 리스트로 저장
        }

    # 각 패턴을 추가할 때 '|'를 제거한 후 추가
    for pattern in patterns:
        intents_dict[tag]['patterns'].append(pattern.strip())  # 공백 제거 후 추가

# intents_list로 변환
intents_list = list(intents_dict.values())

# 최종 JSON 데이터 구조
output_data = {
    "intents": intents_list
}

# JSON 파일로 저장
with open('C:/Users/Admin/Desktop/Campus Helper/json/output.json', 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)

# 데이터베이스 연결 종료
cursor.close()
conn.close()

print("JSON 파일이 성공적으로 생성되었습니다: output.json")
