import fitz 
import re
import os

def pdf_to_txt(pdf_path, output_txt):

    if not os.path.exists(pdf_path):
        print(f"Ошибка: Файл {pdf_path} не найден.")
        return


    doc = fitz.open(pdf_path)
    full_text = ""
    

    print("Извлечение текста из PDF...")
    for page in doc:

        full_text += page.get_text("text") 
    doc.close()
    

    pattern = re.compile(r'(ЧАСТЬ\s+ПЕРВАЯ\s+I\b.*?)(?=\n\s*III\b)', re.DOTALL)
    
    match = pattern.search(full_text)
    
    if match:
        text_chunk = match.group(1)
        

        text_chunk = re.sub(r'\n{3,}', '\n\n', text_chunk)
        
      
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(text_chunk)
            
        print(f"✅ Готово! Текст ЧАСТИ ПЕРВОЙ (главы I и II) успешно сохранен в файл: {output_txt}")
    else:
        print("⚠️ Не удалось найти точные границы глав с помощью регулярного выражения.")
        print("Возможно, в PDF специфическое форматирование (например, номера глав слиты с текстом).")
        print("В качестве отладки сохраняю весь извлеченный текст в 'debug_full_text.txt'...")
        with open("debug_full_text.txt", 'w', encoding='utf-8') as f:
            f.write(full_text)

if __name__ == "__main__":
    INPUT_PDF = "Преступление_и_наказание.pdf"
    OUTPUT_TXT = "Part1_Ch1_2.txt"
    
    pdf_to_txt(INPUT_PDF, OUTPUT_TXT)