def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

# Given a filename as a string, find the timestamp
def find_timestamp(bbox_fname):
  s = str(bbox_fname)
  timestamp = find_between(s, ":", ":")
  return timestamp

def val_append(dict_obj, key, value):
 if key in dict_obj:
  if not isinstance(dict_obj[key], list):
  # converting key to list type
   dict_obj[key] = [dict_obj[key]]
   # Append the key's value in list
   dict_obj[key].append(value)