"""I export one saved Google Doc tab to Markdown without fetching remote content."""

import argparse
import hashlib
import html
import json
import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote


def escape(text):
    text = html.escape(text, quote=False)
    return re.sub(r'([\\`*_\[\]])', r'\\\1', text)


def render_run(text, style):
    text = text.replace('\v', '\n')
    leading, body, trailing = re.fullmatch(r'(\s*)(.*?)(\s*)', text, re.S).groups()
    if not body:
        return text
    value = escape(body)
    if style.get('italic'):
        value = '*' + value + '*'
    if style.get('bold'):
        value = '**' + value + '**'
    if style.get('strikethrough'):
        value = '~~' + value + '~~'
    if style.get('link'):
        value = '[' + value + '](' + style['link'].replace(')', '%29') + ')'
    return leading + value + trailing


def render_image(object_id, images):
    entry = images.get(object_id)
    if not isinstance(entry, dict):
        raise ValueError(f'I need an image mapping for {object_id}.')
    path, alt = entry.get('path'), entry.get('alt')
    if not isinstance(path, str) or not path or not isinstance(alt, str):
        raise ValueError(f'I need string path and alt fields for {object_id}.')
    if PurePosixPath(path).is_absolute() or '\\' in path or re.match(r'^[\w+.-]+:', path):
        raise ValueError(f'I need a relative image path for {object_id}.')
    if any(c in path + alt for c in '\r\n\x00'):
        raise ValueError(f'I cannot render multiline image metadata for {object_id}.')
    return '![' + escape(alt) + '](' + quote(path, safe='/._-~') + ')'


def convert(doc, main_only=False, images=None, tab_id='t.0'):
    """I return Markdown, original text runs and the number of exported paragraphs."""
    matches = [tab for tab in doc.get('tabs', []) if tab.get('tabId') == tab_id]
    if len(matches) != 1:
        raise ValueError(f'I need exactly one saved tab with ID {tab_id}.')
    tab = matches[0]
    if tab.get('positionedObjects') or doc.get('positionedObjects'):
        raise ValueError('I cannot preserve positioned objects in this converter.')
    for part in ('headers', 'footers', 'footnotes'):
        if tab.get(part):
            raise ValueError(f'I cannot preserve {part} in this converter.')
    if images is None:
        images = {}
    if not isinstance(images, dict):
        raise ValueError('I need an image mapping keyed by inline object ID.')
    body = tab.get('body')
    if not isinstance(body, dict) or not isinstance(body.get('content'), list):
        raise ValueError('I need the saved tab body and its content list.')

    blocks, source, list_counters = [], [], {}
    count = 0
    for node in body['content']:
        content_keys = set(node) - {'startIndex', 'endIndex'}
        if content_keys == {'sectionBreak'}:
            continue
        if content_keys != {'paragraph'}:
            raise ValueError(f'I cannot preserve document elements: {sorted(content_keys)}.')
        paragraph = node['paragraph']
        if paragraph.get('positionedObjectIds'):
            raise ValueError('I cannot preserve positioned paragraph objects.')
        elements = paragraph.get('elements', [])
        raw = ''.join(
            element.get('textRun', {}).get('content', '')
            if 'textRun' in element else
            element.get('richLink', {}).get('richLinkProperties', {}).get('title', '')
            for element in elements)
        if main_only and raw.strip() == 'Working notes':
            break
        source.append(raw)
        count += 1
        heading = paragraph.get('paragraphStyle', {}).get('namedStyleType', 'NORMAL_TEXT')
        runs = []
        for element in elements:
            keys = set(element) - {'startIndex', 'endIndex'}
            if keys == {'pageBreak'}:
                continue
            if keys == {'inlineObjectElement'}:
                object_id = element['inlineObjectElement'].get('inlineObjectId')
                obj = (tab.get('inlineObjects') or {}).get(object_id, {})
                embedded = obj.get('inlineObjectProperties', {}).get('embeddedObject', {})
                if 'imageProperties' not in embedded:
                    raise ValueError(f'I cannot preserve the non-image inline object {object_id}.')
                runs.append((render_image(object_id, images), None))
                continue
            if keys == {'richLink'}:
                rich_link = element['richLink']
                properties = rich_link.get('richLinkProperties', {})
                title, uri = properties.get('title'), properties.get('uri')
                if not isinstance(title, str) or not isinstance(uri, str) or not uri:
                    raise ValueError('I need a displayed title and URI for each rich link.')
                text_run = {'content': title, 'textStyle': {
                    **rich_link.get('textStyle', {}), 'link': {'url': uri}}}
            elif keys == {'textRun'}:
                text_run = element['textRun']
            else:
                raise ValueError(f'I cannot preserve paragraph elements: {sorted(keys)}.')
            source_style = text_run.get('textStyle', {})
            style = {key: source_style.get(key, False)
                     for key in ('bold', 'italic', 'strikethrough')}
            link = source_style.get('link') or {}
            if link and not link.get('url'):
                raise ValueError('I cannot preserve a link without a URL.')
            style['link'] = link.get('url')
            if heading.startswith('HEADING_'):
                style['bold'] = False
            value = text_run.get('content', '')
            if runs and runs[-1][1] == style:
                runs[-1] = (runs[-1][0] + value, style)
            else:
                runs.append((value, style))
        rendered = ''.join(render_run(text, style) if style is not None else text
                           for text, style in runs).strip()
        if not rendered:
            continue
        if heading.startswith('HEADING_'):
            if heading not in {f'HEADING_{level}' for level in range(1, 7)}:
                raise ValueError(f'I cannot preserve heading style {heading}.')
            rendered = '#' * int(heading.split('_')[-1]) + ' ' + rendered
        elif heading == 'TITLE':
            rendered = '# ' + rendered
        elif paragraph.get('bullet'):
            bullet = paragraph['bullet']
            key, level = bullet['listId'], bullet.get('nestingLevel', 0)
            levels = (tab.get('lists', {}).get(key, {}).get('listProperties', {})
                      .get('nestingLevels', []))
            if not isinstance(level, int) or level < 0 or level >= len(levels):
                raise ValueError(f'I need list properties for {key} at level {level}.')
            props = levels[level]
            glyph = props.get('glyphType')
            if glyph == 'DECIMAL':
                counter_key = (key, level)
                number = list_counters.get(counter_key, props.get('startNumber', 1))
                list_counters[counter_key] = number + 1
                marker = str(number) + '. '
            elif glyph in (None, 'GLYPH_TYPE_UNSPECIFIED'):
                marker = '- '
            else:
                raise ValueError(f'I cannot preserve {glyph} list numbering in Markdown.')
            lines, indent = rendered.split('\n'), '    ' * level
            rendered = indent + marker + lines[0] + ''.join(
                '\n' + (indent + ' ' * len(marker) + line if line else '')
                for line in lines[1:])
        blocks.append(rendered)
    return '\n\n'.join(blocks) + '\n', ''.join(source), count


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', type=Path,
                        default=root / 'provenance/google-doc-readme/document.json')
    parser.add_argument('output', nargs='?', type=Path, default=root / 'README.md')
    parser.add_argument('--images', type=Path,
                        help='Image mapping; defaults to provenance/google-doc-readme/images.json if present.')
    parser.add_argument('--plain-text', type=Path,
                        help='Optional output for original text runs, without image labels.')
    parser.add_argument('--tab', default='t.0', help='Saved tab ID (default: t.0).')
    parser.add_argument('--main-only', action='store_true',
                        help='Stop before the Working notes heading.')
    args = parser.parse_args(argv)
    try:
        doc = json.loads(args.source.read_text(encoding='utf-8'))
        image_path = args.images or root / 'provenance/google-doc-readme/images.json'
        if args.images is not None or image_path.exists():
            images = json.loads(image_path.read_text(encoding='utf-8'))
        else:
            images = {}
        markdown, plain, count = convert(doc, args.main_only, images, args.tab)
        outputs = [args.output] + ([args.plain_text] if args.plain_text else [])
        if len({path.resolve() for path in outputs}) != len(outputs):
            raise ValueError('I need separate Markdown and plain-text output paths.')
        inputs = {args.source.resolve(), image_path.resolve()}
        for path in outputs:
            if path.resolve() in inputs:
                raise ValueError('I keep source snapshots and image mappings separate from outputs.')
            if not path.parent.is_dir():
                raise ValueError(f'I need the output directory to exist: {path.parent}.')
        args.output.write_text(markdown, encoding='utf-8')
        if args.plain_text:
            args.plain_text.write_text(plain, encoding='utf-8')
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({'paragraphs': count, 'bytes': len(markdown.encode('utf-8')),
                      'source_characters': len(plain),
                      'sha256': hashlib.sha256(markdown.encode('utf-8')).hexdigest()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
