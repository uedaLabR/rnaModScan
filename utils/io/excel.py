"""Write small, native Excel workbooks using only the Python standard library.

Strings are stored as literal text, including strings beginning with '='.
Numeric cells retain their values; the two-decimal display is formatting only.
"""
import math
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED


MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
DOC_RELS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PACKAGE_RELS = 'http://schemas.openxmlformats.org/package/2006/relationships'
CONTENT_TYPES = 'http://schemas.openxmlformats.org/package/2006/content-types'
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'


def column_name(number):
    """Convert a one-based column number to its Excel letter label."""
    result = ''
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def xml_bytes(root):
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def append_row(sheet_data, number, values, header=False):
    attributes = {'r': str(number)}
    if header:
        attributes.update(ht='30', customHeight='1')
    row = ET.SubElement(sheet_data, 'row', attributes)
    for column, value in enumerate(values, 1):
        cell = ET.SubElement(row, 'c', r=column_name(column) + str(number))
        if header:
            cell.set('s', '3')
        if value is None:
            continue
        if isinstance(value, bool):
            cell.set('t', 'b')
            ET.SubElement(cell, 'v').text = '1' if value else '0'
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('Excel numeric cells must be finite')
            cell.set('s', '2' if isinstance(value, float) else '1')
            ET.SubElement(cell, 'v').text = str(value)
        elif isinstance(value, str):
            if any(ord(c) < 32 and c not in '\t\n\r' for c in value):
                raise ValueError('Excel strings cannot contain XML control characters')
            cell.set('t', 'inlineStr')
            text = ET.SubElement(ET.SubElement(cell, 'is'), 't')
            text.set(XML_SPACE, 'preserve')
            text.text = value
        else:
            raise TypeError('Use only strings, integers, floats, booleans or None')


def worksheet_xml(sheet):
    headers, rows = sheet['headers'], sheet['rows']
    if not headers or any(not isinstance(value, str) for value in headers):
        raise ValueError('Each sheet requires nonempty string headers')
    if len(headers) > 16384 or len(rows) > 1048575:
        raise ValueError('Sheet exceeds Excel row or column limits')
    if any(len(row) != len(headers) for row in rows):
        raise ValueError('Every row must have the same length as its headers')
    widths = sheet.get('widths', [18.0] * len(headers))
    if len(widths) != len(headers) or any(not 0 < float(w) <= 255 for w in widths):
        raise ValueError('Provide one column width between zero and 255 per header')
    root = ET.Element('worksheet', xmlns=MAIN)
    last_cell = column_name(len(headers)) + str(len(rows) + 1)
    ET.SubElement(root, 'dimension', ref='A1:' + last_cell)
    view = ET.SubElement(ET.SubElement(root, 'sheetViews'), 'sheetView', workbookViewId='0')
    ET.SubElement(view, 'pane', ySplit='1', topLeftCell='A2', activePane='bottomLeft', state='frozen')
    ET.SubElement(view, 'selection', pane='bottomLeft', activeCell='A2', sqref='A2')
    ET.SubElement(root, 'sheetFormatPr', defaultRowHeight='15')
    columns = ET.SubElement(root, 'cols')
    for number, width in enumerate(widths, 1):
        ET.SubElement(columns, 'col', min=str(number), max=str(number),
                      width=str(float(width)), customWidth='1')
    data = ET.SubElement(root, 'sheetData')
    append_row(data, 1, headers, header=True)
    for number, values in enumerate(rows, 2):
        append_row(data, number, values)
    ET.SubElement(root, 'autoFilter', ref='A1:' + last_cell)
    return xml_bytes(root)


def styles_xml(float_format='0.00'):
    root = ET.Element('styleSheet', xmlns=MAIN)
    formats = ET.SubElement(root, 'numFmts', count='1')
    ET.SubElement(formats, 'numFmt', numFmtId='164', formatCode=float_format)
    fonts = ET.SubElement(root, 'fonts', count='2')
    for bold in (False, True):
        font = ET.SubElement(fonts, 'font')
        ET.SubElement(font, 'sz', val='11')
        ET.SubElement(font, 'name', val='Calibri')
        if bold:
            ET.SubElement(font, 'b')
    fills = ET.SubElement(root, 'fills', count='3')
    for pattern in ('none', 'gray125', 'solid'):
        fill = ET.SubElement(ET.SubElement(fills, 'fill'), 'patternFill', patternType=pattern)
        if pattern == 'solid':
            ET.SubElement(fill, 'fgColor', rgb='FFE5EFF8')
            ET.SubElement(fill, 'bgColor', indexed='64')
    border = ET.SubElement(ET.SubElement(root, 'borders', count='1'), 'border')
    for edge in ('left', 'right', 'top', 'bottom', 'diagonal'):
        ET.SubElement(border, edge)
    style_xfs = ET.SubElement(root, 'cellStyleXfs', count='1')
    ET.SubElement(style_xfs, 'xf', numFmtId='0', fontId='0', fillId='0', borderId='0')
    cell_xfs = ET.SubElement(root, 'cellXfs', count='4')
    # Integer thousands separator and a caller-selected floating-point display.
    for number_format, font, fill in ((0, 0, 0), (3, 0, 0), (164, 0, 0), (0, 1, 2)):
        xf = ET.SubElement(cell_xfs, 'xf', numFmtId=str(number_format), fontId=str(font),
                           fillId=str(fill), borderId='0', xfId='0', applyAlignment='1')
        if number_format:
            xf.set('applyNumberFormat', '1')
        if font:
            xf.set('applyFont', '1')
        if fill:
            xf.set('applyFill', '1')
        ET.SubElement(xf, 'alignment', vertical='top', wrapText='1')
    ET.SubElement(ET.SubElement(root, 'cellStyles', count='1'), 'cellStyle',
                  name='Normal', xfId='0', builtinId='0')
    return xml_bytes(root)


def write_workbook(path, sheets, *, float_format='0.00'):
    """Write sheets [{name, headers, rows, widths?}] without overwriting a file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not sheets:
        raise ValueError('At least one worksheet is required')
    names = [sheet['name'] for sheet in sheets]
    for name in names:
        if (not isinstance(name, str) or not name or len(name) > 31
                or re.search(r'[\[\]:*?/\\]', name) or name.startswith("'") or name.endswith("'")):
            raise ValueError('Invalid Excel worksheet name')
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError('Worksheet names must be unique, ignoring case')
    # Serialize first, so invalid values cannot leave a half-written workbook.
    worksheets = [worksheet_xml(sheet) for sheet in sheets]
    workbook = ET.Element('workbook', {'xmlns': MAIN, 'xmlns:r': DOC_RELS})
    ET.SubElement(ET.SubElement(workbook, 'bookViews'), 'workbookView')
    sheet_list = ET.SubElement(workbook, 'sheets')
    relations = ET.Element('Relationships', xmlns=PACKAGE_RELS)
    types = ET.Element('Types', xmlns=CONTENT_TYPES)
    ET.SubElement(types, 'Default', Extension='rels', ContentType='application/vnd.openxmlformats-package.relationships+xml')
    ET.SubElement(types, 'Default', Extension='xml', ContentType='application/xml')
    for part, content_type in (('workbook.xml', 'sheet.main'), ('styles.xml', 'styles')):
        ET.SubElement(types, 'Override', PartName='/xl/' + part,
                      ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.' + content_type + '+xml')
    for number, name in enumerate(names, 1):
        relation_id = 'rId' + str(number)
        ET.SubElement(sheet_list, 'sheet', {'name': name, 'sheetId': str(number), 'r:id': relation_id})
        ET.SubElement(relations, 'Relationship', Id=relation_id,
                      Type=DOC_RELS + '/worksheet', Target=f'worksheets/sheet{number}.xml')
        ET.SubElement(types, 'Override', PartName=f'/xl/worksheets/sheet{number}.xml',
                      ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')
    ET.SubElement(relations, 'Relationship', Id='rId' + str(len(names) + 1),
                  Type=DOC_RELS + '/styles', Target='styles.xml')
    package = ET.Element('Relationships', xmlns=PACKAGE_RELS)
    ET.SubElement(package, 'Relationship', Id='rId1', Type=DOC_RELS + '/officeDocument', Target='xl/workbook.xml')
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, 'x', ZIP_DEFLATED) as archive:
        for name, content in (('[Content_Types].xml', xml_bytes(types)),
                              ('_rels/.rels', xml_bytes(package)), ('xl/workbook.xml', xml_bytes(workbook)),
                              ('xl/_rels/workbook.xml.rels', xml_bytes(relations)), ('xl/styles.xml', styles_xml(float_format))):
            archive.writestr(name, content)
        for number, content in enumerate(worksheets, 1):
            archive.writestr(f'xl/worksheets/sheet{number}.xml', content)


def verify_workbook(path, *, sheet_names=None):
    """Read this writer's native cells back as {sheet_name: [headers, *rows]}.

    Checks ZIP integrity, numeric/text cell types, frozen headers and filters.
    This is a structural check, not a spreadsheet-application rendering test.
    """
    ns = {'s': MAIN}
    selected = None if sheet_names is None else set(sheet_names)
    restored = {}
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Workbook ZIP integrity check failed')
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        relations = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets = {r.get('Id'): r.get('Target') for r in relations}
        ET.fromstring(archive.read('xl/styles.xml'))
        for sheet in workbook.findall('s:sheets/s:sheet', ns):
            if selected is not None and sheet.get('name') not in selected:
                continue
            target = targets[sheet.get('{' + DOC_RELS + '}id')]
            root = ET.fromstring(archive.read('xl/' + target))
            pane = root.find('s:sheetViews/s:sheetView/s:pane', ns)
            if pane is None or pane.get('state') != 'frozen' or root.find('s:autoFilter', ns) is None:
                raise ValueError('Worksheet lacks frozen headers or filter')
            rows = []
            for row in root.findall('s:sheetData/s:row', ns):
                values = []
                for cell in row.findall('s:c', ns):
                    kind = cell.get('t')
                    value = cell.find('s:v', ns)
                    if kind == 'inlineStr':
                        text = cell.find('s:is/s:t', ns)
                        values.append(text.text or '' if text is not None else '')
                    elif value is None:
                        values.append(None)
                    elif kind == 'b':
                        values.append(value.text == '1')
                    elif cell.get('s') == '2':
                        values.append(float(value.text))
                    else:
                        values.append(int(value.text))
                rows.append(values)
            restored[sheet.get('name')] = rows
    if selected is not None and set(restored) != selected:
        raise ValueError('Requested Excel sheets are missing: ' + ', '.join(sorted(selected - set(restored))))
    return restored
