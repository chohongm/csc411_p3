<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

if(!isset($_SESSION['expandlist']))  {
	$_SESSION['expandlist'] = array();
}

#when user clicks on expand (+) button.
if(isset($_GET['expand']))   {
	if($_GET['expand'] == 'all') {	
		#expand all threads stored in expanded list.
		$_SESSION['expandlist'] = get_expand_all_list();
	} else {
		#expand the thread of interest by appending it to expandlist with true being its value.
		$_SESSION['expandlist'][$_GET['expand']] = true;
	}
}

#when user clicks on collapse (-) button.
if(isset($_GET['collapse'])) {
	if($_GET['collapse']=='all') {
		#empty expanded list.
		$_SESSION['expandlist'] = array();
	} else {
		#unset a post in expanded list.
		unset($_SESSION['expandlist'][$_GET['collapse']]);
	}
}

do_forum_header('Forum Posts');
display_index_toolbar();
display_tree($_SESSION['expandlist']);

do_footer();
?>