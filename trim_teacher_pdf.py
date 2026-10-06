from pypdf import PdfReader, PdfWriter

src = "/Users/botlv/CUP-code/cup_comparison_tables_for_teacher.pdf"
reader = PdfReader(src)
writer = PdfWriter()
for page in reader.pages[:-1]:
    writer.add_page(page)
with open(src, "wb") as stream:
    writer.write(stream)
print(f"Kept {len(writer.pages)} pages")
