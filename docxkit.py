"""Utilidades para llenar la plantilla IEEE (A4) con python-docx."""
import copy
import re
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


class Reporte:
    def __init__(self, plantilla):
        self.d = Document(plantilla)
        self.S = {st.name: st for st in self.d.styles}
        body = self.d.element.body
        hijos = list(body)
        # 0 título, 1 nota subtítulo, 2 vacío, 3 corte, 4-9 autores, 10 corte, 11 Normal, 12 corte
        self.titulo_p = hijos[0]
        self.autores_p = hijos[4:10]
        body.remove(hijos[1])
        for el in hijos[13:]:
            if el.tag != qn("w:sectPr"):
                body.remove(el)
        final = body.find(qn("w:sectPr"))
        cols = final.find(qn("w:cols"))
        if cols is None:
            cols = OxmlElement("w:cols"); final.append(cols)
        cols.set(qn("w:num"), "2"); cols.set(qn("w:space"), "360")
        typ = final.find(qn("w:type"))
        if typ is None:
            typ = OxmlElement("w:type"); final.insert(0, typ)
        typ.set(qn("w:val"), "continuous")

    # ------------------------------------------------------------------ encabezado
    def titulo(self, texto):
        p = self.titulo_p
        for r in p.findall(qn("w:r")):
            p.remove(r)
        self._run(p, texto)

    def autores(self, lista):
        """Autores en orden de lectura, repartidos 2 | 2 | 1 en las tres columnas con saltos de columna."""
        for k, p in enumerate(self.autores_p):
            for r in p.findall(qn("w:r")):
                p.remove(r)
            if k >= len(lista):
                p.getparent().remove(p)
                continue
            for i, linea in enumerate(lista[k]):
                if i:
                    r = OxmlElement("w:r"); r.append(OxmlElement("w:br")); p.append(r)
                self._run(p, linea, italic=(i in (1, 2)))
            if k in (1, 3) and k < len(lista) - 1:
                r = OxmlElement("w:r"); br = OxmlElement("w:br"); br.set(qn("w:type"), "column")
                r.append(br); p.append(r)

    @staticmethod
    def _run(p, texto, italic=False, bold=False):
        r = OxmlElement("w:r")
        if italic or bold:
            rpr = OxmlElement("w:rPr")
            if bold: rpr.append(OxmlElement("w:b"))
            if italic: rpr.append(OxmlElement("w:i"))
            r.append(rpr)
        t = OxmlElement("w:t"); t.set(qn("xml:space"), "preserve"); t.text = texto
        r.append(t); p.append(r)

    # ------------------------------------------------------------------ cuerpo
    def par(self, texto, estilo="Body Text", align=None):
        """Texto con marcado ligero: *itálica*, **negrita**, _{sub}, ^{sup}."""
        p = self.d.add_paragraph(style=self.S[estilo])
        self._rich(p, texto)
        if align == "center":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return p

    def _rich(self, p, texto):
        tokens = re.split(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})", texto)
        for tk in tokens:
            if not tk:
                continue
            if tk.startswith("**"):
                p.add_run(tk[2:-2]).bold = True
            elif tk.startswith("*"):
                p.add_run(tk[1:-1]).italic = True
            elif tk.startswith("_{"):
                p.add_run(tk[2:-1]).font.subscript = True
            elif tk.startswith("^{"):
                p.add_run(tk[2:-1]).font.superscript = True
            else:
                p.add_run(tk)

    def abstract(self, texto):
        p = self.d.add_paragraph(style=self.S["Abstract"])
        r = p.add_run("Resumen—"); r.italic = True; r.bold = True
        self._rich(p, texto)

    def keywords(self, texto):
        p = self.d.add_paragraph(style=self.S["Keywords"])
        r = p.add_run("Palabras clave—"); r.italic = True; r.bold = True
        p.add_run(texto)

    def h1(self, t): return self.d.add_paragraph(t, style=self.S["Heading 1"])
    def h2(self, t): return self.d.add_paragraph(t, style=self.S["Heading 2"])
    def h5(self, t): return self.d.add_paragraph(t, style=self.S["Heading 5"])

    def viñeta(self, texto):
        return self.par(texto, "bullet list")

    def ecuacion(self, texto, num):
        p = self.d.add_paragraph(style=self.S["equation"])
        p.add_run("\t")
        self._rich(p, texto)
        p.add_run(f"\t({num})")
        for r in p.runs:
            rf = r._r.get_or_add_rPr().get_or_add_rFonts()
            for k in ["w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"]:
                rf.set(qn(k), "Times New Roman")
        return p

    def figura(self, ruta, pie, ancho=3.4):
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(ruta), width=Inches(ancho))
        return self.par(pie, "figure caption")

    def tabla(self, titulo, cabecera, filas, anchos, nota=None, font=7.5):
        cap = self.d.add_paragraph(titulo, style=self.S["table head"])
        cap.paragraph_format.keep_with_next = True
        tot = sum(anchos)
        anchos = [a * 3.4 / tot for a in anchos]
        t = self.d.add_table(rows=1 + len(filas), cols=len(cabecera))
        grid = t._tbl.tblGrid
        for gc, a in zip(grid.findall(qn("w:gridCol")), anchos):
            gc.set(qn("w:w"), str(int(a * 1440)))
        t.alignment = 1
        tblPr = t._tbl.tblPr
        bordes = OxmlElement("w:tblBorders")
        for lado in ["top", "left", "bottom", "right", "insideH", "insideV"]:
            b = OxmlElement(f"w:{lado}")
            b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "4"); b.set(qn("w:color"), "000000")
            bordes.append(b)
        tblPr.append(bordes)
        tw = tblPr.find(qn("w:tblW"))
        if tw is None:
            tw = OxmlElement("w:tblW"); tblPr.append(tw)
        tw.set(qn("w:type"), "dxa"); tw.set(qn("w:w"), str(int(3.4 * 1440)))
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
        for i, fila in enumerate([cabecera] + filas):
            for j, val in enumerate(fila):
                c = t.cell(i, j)
                c.width = Inches(anchos[j])
                p = c.paragraphs[0]
                p.style = self.S["table col head" if i == 0 else "table copy"]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                self._rich(p, str(val))
                for r in p.runs:
                    r.font.size = Pt(font)
                    if i == 0:
                        r.bold = True
                pf = p.paragraph_format
                pf.space_before = Pt(0); pf.space_after = Pt(0)
                if i == 0:
                    tr = c._tc.getparent()
                    trPr = tr.get_or_add_trPr()
                    h = OxmlElement("w:tblHeader"); trPr.append(h)
        if nota:
            self.par(nota, "table footnote")
        else:
            sp = self.d.add_paragraph(style=self.S["Normal"])
            sp.paragraph_format.space_after = Pt(2)
        return t

    def referencia(self, texto):
        return self.par(texto, "references")

    def pie_de_pagina(self, texto):
        for s in self.d.sections:
            for f in [s.footer, s.first_page_footer]:
                for p in f.paragraphs:
                    if "XXX" in p.text or "IEEE" in p.text:
                        for r in p.runs:
                            r.text = ""
                        if p.runs:
                            p.runs[0].text = texto

    def guardar(self, ruta):
        num = self.d.part.numbering_part.element
        for lt in num.iter(qn("w:lvlText")):
            v = lt.get(qn("w:val"))
            if v and "TABLE" in v:
                lt.set(qn("w:val"), v.replace("TABLE", "TABLA"))
        self.d.save(ruta)
