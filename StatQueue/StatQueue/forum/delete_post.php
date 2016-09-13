<?php

$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
session_start();

if (isset($_GET['delete'])){
	$post = $_SESSION['post'];
	$value = delete_post($post);
	if ($value == 'all'){
		$_GET['expand'] = 'all';
		include("$DOCUMENT_ROOT/forum/index.php");
	} else {
		$_GET['postid'] = $value;
		include("$DOCUMENT_ROOT/forum/view_post.php");
	}
}

?>