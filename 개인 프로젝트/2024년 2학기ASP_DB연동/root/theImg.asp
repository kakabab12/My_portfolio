<%
' ------------------------------------------------------------------------------------------
' theImg.asp
'
' To retrieve the content of the image.  There's no way to avoid creating a proxy page when
' pulling an image from a database.  However, this proxy page, designed by Dino Esposito (see
' comment in BinFile1), accesses an existing recordset instead of executing a brand new query. 
' ------------------------------------------------------------------------------------------

%>
<% 
    response.Expires = 0
    response.Buffer  = True
    response.Clear
   
    response.contentType = Session("ImageType")
    response.BinaryWrite Session("ImageBytes")

    Session("ImageType") = ""
    Session("ImageBytes") = ""

    response.End
%>