/**
 * @license
 * Copyright 2019 Google LLC.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * =============================================
 */

// unicode-aware iteration
export const stringToChars = (input: string): string[] => {
  const symbols = [];
  for (const symbol of input) {
    symbols.push(symbol);
  }
  return symbols;
};
