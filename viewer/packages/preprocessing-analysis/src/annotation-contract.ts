// These values mirror dr_code.viewer.domain and are pinned cross-language.
// Do not replace the whitespace literals with runtime-specific predicates.
export const ANNOTATION_NOTE_MAX_LENGTH = 10_000;
export const ANNOTATION_TAG_IDS_MAX_COUNT = 100;
export const TAG_NAME_MAX_LENGTH = 100;
export const TAG_NAME_WHITESPACE_CODE_POINTS = [
  0x0009,
  0x000a,
  0x000b,
  0x000c,
  0x000d,
  0x0020,
  0x0085,
  0x00a0,
  0x1680,
  0x2000,
  0x2001,
  0x2002,
  0x2003,
  0x2004,
  0x2005,
  0x2006,
  0x2007,
  0x2008,
  0x2009,
  0x200a,
  0x2028,
  0x2029,
  0x202f,
  0x205f,
  0x3000,
] as const;

const TAG_NAME_WHITESPACE = new Set<number>(TAG_NAME_WHITESPACE_CODE_POINTS);

export function isUnicodeScalarText(value: string): boolean {
  return Array.from(value).every((character) => {
    const codePoint = character.codePointAt(0);
    return codePoint !== undefined && (codePoint < 0xd800 || codePoint > 0xdfff);
  });
}

export function contractLength(value: string): number {
  return Array.from(value).length;
}

export function normalizeTagName(value: string): string {
  const normalizedCharacters = Array.from(value, (character) => {
    const codePoint = character.codePointAt(0);
    return codePoint !== undefined && TAG_NAME_WHITESPACE.has(codePoint) ? " " : character;
  });
  return normalizedCharacters.join("").split(" ").filter(Boolean).join(" ");
}

export function isTagNameInContract(value: string): boolean {
  if (!isUnicodeScalarText(value)) return false;
  const normalized = normalizeTagName(value);
  const length = contractLength(normalized);
  return length > 0 && length <= TAG_NAME_MAX_LENGTH;
}

export function isAnnotationNoteInContract(value: string): boolean {
  return isUnicodeScalarText(value) && contractLength(value) <= ANNOTATION_NOTE_MAX_LENGTH;
}
