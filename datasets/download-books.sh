#!/bin/bash
# This script downloads UTF‑8 text files for a hardcoded list of Project Gutenberg books.
# It uses common download patterns and saves the files using the book title as the filename,
# after sanitizing the title for safe filesystem use. Downloads are saved in the
# "public_book_texts" directory.
#
# If the primary URL (with "-0.txt") fails, it falls back to the URL without "-0.txt".

# Define the list of books as "ID,Title" entries.
book_list=(
"3300,Wealth of Nations"
"2701,Moby Dick"
"730,Oliver Twist"
"1342,Pride and Prejudice"
"1661,The Adventures of Sherlock Holmes"
"12,Through The Looking Glass"
"1998,Thus Spake Zarathustra"
"100,Complete Works of Shakespeare"
"75543,Illustrated Commentary on the Gospel"
)

# Create the download directory if it does not already exist.
download_dir="public_book_texts"
mkdir -p "$download_dir"

# Process each book entry.
for entry in "${book_list[@]}"; do
    # Split the entry into ID and title using a comma as the delimiter.
    IFS=',' read -r id title <<< "$entry"

    # Sanitize the title to create a safe filename:
    # - Transliterate non-ASCII characters to ASCII.
    # - Remove characters other than letters, numbers, space, underscore, or dash.
    # - Replace spaces with underscores.
    safe_title=$(echo "$title" | iconv -t ascii//TRANSLIT | sed 's/[^a-zA-Z0-9 _-]//g' | tr ' ' '_')

    # Define the primary download URL pattern.
    url="https://www.gutenberg.org/files/${id}/${id}-0.txt"
    echo "Attempting to download '$title' (ID: $id) from:"
    echo "  $url"

    # Try downloading using curl. The -f flag causes curl to fail on HTTP errors.
    if ! curl -f -s "$url" -o "$download_dir/${safe_title}.txt"; then
        # If the primary URL fails, try the fallback URL pattern.
        fallback_url="https://www.gutenberg.org/files/${id}/${id}.txt"
        echo "Primary URL failed; trying fallback URL:"
        echo "  $fallback_url"
        if ! curl -f -s "$fallback_url" -o "$download_dir/${safe_title}.txt"; then
            echo "Failed to download '$title' (ID: $id)."
            # Remove any (possibly partial) file.
            rm -f "$download_dir/${safe_title}.txt"
        else
            echo "Downloaded '$title' successfully using fallback URL."
        fi
    else
        echo "Downloaded '$title' successfully."
    fi
    echo "--------------------------------------------"
done

echo "All downloads complete. Files are saved in the '$download_dir' directory."
