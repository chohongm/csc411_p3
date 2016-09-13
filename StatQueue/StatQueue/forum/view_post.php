<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

$postid = $_GET['postid'];
$post = get_post($postid);
$post['dateposted'] = reformat_date($post['dateposted']);
do_forum_header($post['title']);

display_post($post);
#if post has any reply, show them.
if ($post['children']){
	echo '<br/><br/>
		  <table width = 1000 cellpadding = "4" cellspacing = "0" bgcolor = "#cccccc">
		  <tr><td><strong>Replies to this message</strong></td></tr>
		  </table>';
	display_tree($_SESSION['expandlist'], $postid);
}

do_footer();

?>