import fitz  # PyMuPDF
import re
import os

def pdf_to_txt(pdf_path, output_txt):
    # Проверяем, существует ли файл
    if not os.path.exists(pdf_path):
        print(f"Ошибка: Файл {pdf_path} не найден.")
        return

    # Открываем PDF
    doc = fitz.open(pdf_path)
    full_text = ""
    
    # Извлекаем только текст (картинки и прочее автоматически игнорируются)
    print("Извлечение текста из PDF...")
    for page in doc:
        # get_text("text") возвращает только текстовые блоки
        full_text += page.get_text("text") 
    doc.close()
    
    # Регулярное выражение для поиска нужного фрагмента:
    # Начинаем с "ЧАСТЬ ПЕРВАЯ" и главы "I"
    # Заканчиваем перед началом главы "III"
    # re.DOTALL позволяет точке захватывать символы переноса строки
    pattern = re.compile(r'(ЧАСТЬ\s+ПЕРВАЯ\s+I\b.*?)(?=\n\s*III\b)', re.DOTALL)
    
    match = pattern.search(full_text)
    
    if match:
        text_chunk = match.group(1)
        
        # Очищаем текст от множественных пустых строк (более 2-х подряд)
        text_chunk = re.sub(r'\n{3,}', '\n\n', text_chunk)
        
        # Сохраняем в .txt файл
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
    # Укажите имя вашего PDF файла и желаемое имя выходного TXT файла
    INPUT_PDF = "Преступление_и_наказание.pdf"
    OUTPUT_TXT = "Part1_Ch1_2.txt"
    
    pdf_to_txt(INPUT_PDF, OUTPUT_TXT)