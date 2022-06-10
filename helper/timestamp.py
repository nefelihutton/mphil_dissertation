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
