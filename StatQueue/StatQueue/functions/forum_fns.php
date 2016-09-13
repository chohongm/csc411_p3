<?php

require_once('db_fns.php');
require_once('output_fns.php');

$table_width = 1000;

function do_forum_header($title = '') {
// print an forum header including cute logo :)
	global $table_width;
?>    
  <table width=<?php echo $table_width; ?> cellspacing="0" cellpadding="6">
  	<tr>
 	  <td bgcolor="#0b215b" width="80" align = center><img src="/images/forum.png"
 	  	  width="60" height="60" alt="" valign="middle" /></td>
  	  <td bgcolor="#0b215b"><t1><?php echo $title; ?></t1></td>
  	</tr>
  </table>
<?php
}

function display_index_toolbar() {
	global $table_width;
?>
  <table width="<?php echo $table_width; ?>" cellpadding="4" cellspacing="0">
    <tr>
	  <td bgcolor="#cccccc" align="right">
        <a href="post_form.php?parent=0"><img src="/images/newpost.png" 
           border="0" width="100" height="30"></a>
        <a href="index.php?expand=all"><img src="/images/expand.png" 
           border="0" width="85" height="30" alt="Expand All Threads"></a>
		<a href="index.php?collapse=all"><img src="/images/collapse.png"
           border="0" width="90" height="30" alt="Collapse All Threads"></a>
      </td>
    </tr>
  </table>
<?php
}

function display_post($post){
	global $table_width;
?>
	<table width = <?php echo $table_width; ?> cellpadding = 4 cellspacing = 0>
	<tr>
	  <td bgcolor = "#cccccc">
	  <strong>from: <?php echo $post['poster'];?><br/> posted: <?php echo $post['dateposted'];?></strong>
	  </td>
	  
	  <td bgcolor = "#ccccc" align = "right">
	  
	  <?php 
	  if (isset($_SESSION['userid'])){
		  if($post['poster'] == $_SESSION['userid']){
		  	$_SESSION['post'] = $post;
		  	if ($post['children'] == 0){
		  		echo '<a href="delete_post.php?delete=true"
					   onclick="return confirm(\'Are you sure you wish to delete this post?\')">
					   <img src="/images/delete.png" border="0" width="24" height="24" /></a> ';
		  	}
		  		echo '<a href="post_form.php?edit=true"><img src="/images/edit.png" border="0" width="24" height="24" /></a>';
		  }
	  }
	  ?>
	  <a href="post_form.php?parent=0">
	  <img src="/images/newpost.png" border="0" width="100" height="30" /></a>
	  <a href="post_form.php?parent=<?php echo $post['postid']; ?>">
	  <img src="/images/reply.png" border="0" width="75" height="30" /></a>
	  <a href="index.php?expanded=<?php echo $post['postid']; ?>">
	  <img src="/images/index.png" border="0" width="75" height="30" /></a>
	  </td>
    </tr>
    
    <tr><td colspan="2"><?php echo nl2br($post['message']);?></td></tr>
	</table>
<?php
}

function display_post_form($parent, $poster, $title, $message, $edit = false){
	global $table_width;
	if ($edit != false){
		$url = 'edit_post.php';
		$postid_input = '<input type="hidden" name="postid" value="'.$edit.'">';
	} else {
		$url = 'post_new.php';
	}
?>
	<table cellpadding="0" cellspacing="0" border="0" width="<?php echo $table_width; ?>">
	<form action="<?php echo $url; ?>?expand=<?php echo $parent;?>#<?php echo $parent;?>" method="post">
	  <tr height=30>
		<td style="text-align: right" width=5% bgcolor="#cccccc">User ID:</td>
		<td bgcolor="#cccccc"><?php echo $poster; ?></td>
		<input type="hidden" name="poster" value="<?php echo $poster; ?>">
	  </tr>
	  <tr height=30>
		<td style="text-align: right" width=5% bgcolor="#cccccc">Title:</td>
		<td bgcolor="#cccccc"><input type="text" name="title" value="<?php echo $title; ?>" 
			size="40" maxlength="40" /></td>
	  </tr>
	  <tr>
		<td colspan="2"><textarea name="message" rows="15" cols="80"><?php echo stripslashes($message);?>
		</textarea></td>
	  </tr>
	  <tr>
	    <td height=45 colspan="2" align="center" bgcolor="#cccccc">
	    	<input type="hidden" name="parent" value="<?php echo $parent; ?>">
	    	<input type="hidden" name="area" value="<?php echo $area; ?>">
	    	<?php if (isset($postid_input)){ echo $postid_input; }?>
	    	<input type="image" name="post" src="/images/post.gif" alt="Post Message" width="75" height="30"/>
	    </td>
	  </tr>
	  </form>
	  </table>
	<?php
}

function store_new_post($post){
	$db = db_connect();
	
	$post = clean_all($post);
	
	#stores new post (reply post containing parent post) into database.
	if(!check_if_duplicate($db, $post)){
		#but first, make sure the post is not already inserted into db.

		if ($post['parent'] != 0){
			#If parentid is specified, check if specified parent post exists in db.
			if (lookup_parent($db, $post['parent']) == false){
				throw new Exception('Specified parentid does not exist in database.');
			}
			#If specified parent exists, insert the post.
			$query = "insert into header values
			  (null, '".$post['poster']."', ".$post['parent'].", 0, '".$post['title']."', now())";
			$result = $db->query($query);
				
			#Inserted post is now child of the specified parent. set children to true.
			$query = "update header set children = 1 where postid = ".$post['parent'];
			$result = $db->query($query);

		} else {
			#if parent is not specified, post is a new thread. Insert.
			$query = "insert into header values
			  (null, '".$post['poster']."', ".$post['parent'].", 0, '".$post['title']."', now())";
			$result = $db->query($query);
		}

		/*retrieve the inserted post's postid.
		 To be exact, specify prentid, title, poster, and matching empty body*/
		$query = "select h.postid from header as h left join body as b on h.postid = b.postid
				  where parent = ".$post['parent']."
				  and title = '".$post['title']."'
				  and poster = '".$post['poster']."'
				  and b.postid is NULL";
				  $result = $db->query($query);
				  $row = $result->fetch_array();
				  $postid = $row[0];

				  #finally, insert message to body.
		$query = "insert into body values (".$postid.", '".$post['message']."')";
		$result = $db->query($query);
		return true;
	}
}

function edit_post($post){
	$db = db_connect();
	$post = clean_all($post);
	$query = "update header set title = '".$post['title']."' where postid = ".$post['postid'];
	$result = $db->query($query);
	$query = "update body set message = '".$post['message']."' where postid = ".$post['postid'];
	$result = $db->query($query);
	return true;
}

function lookup_parent($db, $parent){
	if ($parent != 0){
		$query = "select postid from header where postid = ".$parent;
		$result = $db->query($query);
		if ($result->num_rows > 0){
			return $result->fetch_row()[0];
		}
		return false;
	}
}

function check_if_duplicate($db, $post){	
	$query = "select h.postid from header as h left join body as b
			  on h.postid = b.postid
			  where h.parent = ".$post['parent']."
			  and h.poster = '".$post['poster']."'
			  and h.title = '".$post['title']."'
			  and b.message = '".$post['message']."'";
	$result = $db->query($query);
	if ($result->num_rows != 0){
		return true;
	}
	return false;
}

function delete_post($post){
	$db = db_connect();
	$query = "delete from header where postid = ".$post['postid'];
	$result = $db->query($query);
	$query = "delete from body where postid = ".$post['postid'];
	$result = $db->query($query);
	if (!$result){
		throw new Exception("Post could not be deleted.");
	}
	
	#if the deleted post was a reply post, see if its parent post still has a reply.
	if ($post['parent'] != 0){
		$query = "select * from header where parent = ".$post['parent'];
		$result = $db->query($query);
		
		#if the parent post no longer has a reply, set its children to 0 (false).
		if ($result->num_rows == 0){
			$query = "update header set children = 0 where postid = ".$post['parent'];
			$result = $db->query($query);
		}
		return $post['parent'];
	} else {
		return 'all';
	}
}

function get_expand_all_list(){
	#to expand every post with child post.	
	$db = db_connect();
	$query = "select postid from header where children = 1";
	$result = $db->query($query);
	for($i = 0; $i < $result->num_rows; $i++){
		$row = $result->fetch_row();
		$expandlist[$row[0]] = true;
	}
	return $expandlist;
}

function get_post($postid){
	$db = db_connect();
	#return a list containing given postid's header and body.
	$query = "select * from header left join body on header.postid = body.postid
				  where header.postid = ".$postid;
	$result = mysqli_query($db, $query);
	if ($result->num_rows > 0){
		$post = $result->fetch_assoc();
		return $post;
	}
}

function get_post_title($postid){
	#return title of given post.
	$db = db_connect();
	$query = "select title from header where postid = '".$postid."'";
	$result = $db->query($query);
	if ($result->num_rows > 0){
		$row = $result->fetch_array();
		return $row[0];
	}
}

function get_post_message($postid){
	#return message of given post.
	$db = db_connect();
	$query = "select message from body where postid = ".$postid;
	$result = $db->query($query);
	if ($result->num_rows > 0){
		$row = $result->fetch_array();
		return $row[0];
	}
}

function add_quoting($string, $pattern = '> '){
	#place '>' at the start of given new line. (indicates message is of reply post)
	return $pattern.str_replace("\n", "\n$pattern", $string);
}

function reformat_date($datetime){
	list($year, $month, $day, $hour, $min, $sec) = preg_split('/[\s:-]+/', $datetime);
	return "".$hour.":".$min." ".$month."/".$day."/".$year;
}

class treenode {

	public $db, $postid, $children, $title, $dateposted, $depth, $child_tns;
	
	#Purpose of this class is to recursively build a complete tree using the first instance as root.

	function __construct($tn_info, $is_sublist, $expand, $expandlist, $depth){
		
		/* steps
		 * 1. get all children
		 * 2. check each on expandlist
		 * 3. set appropriate expand state
		 * 4. create tn instance of each child
		 * 5. add to child_tns list
		 * 
		 * note that root treenode is not the interest here. However, display_header recursion
		 * starts at and includes root.
		 */
		$this->db = db_connect();
		$this->postid = $tn_info['postid'];
		$this->poster = $tn_info['poster'];
		$this->children = $tn_info['children'];
		$this->title = $tn_info['title'];
		$this->dateposted = $tn_info['dateposted'];
		$this->child_tns = array();
		$this->depth = $depth;
		
/* 		#testing comments
		if ($is_sublist == false){
			$sublist = 'false';
		} else {
			$sublist = 'true';
		}
		if ($expand == true){
		 $expandd = 'true';
		 } else {
		 $expandd = 'false';
		 }
		echo "<br />postid: ".$this->postid." depth: ".$this->depth." is_sublist:
			 ".$sublist." children: ".$this->children." num_children: ".count($this->child_tns)." expand: 
			 ".$expandd."<br/>"; */
		
		/*if this node is a sublist or a main post that is set to expand and has children,
		  retrieve all its child posts.*/
		if (($is_sublist || $expand) && $this->children){
			$query = "select * from header where parent = ".$this->postid;
			$result = $this->db->query($query);
			
			/*for each child post, check if its listed on expandlist and set expand to true if it is.
			 Then add it to child treenodes list.*/
			for ($count = 0; $child_tn_info = @$result->fetch_assoc(); $count++){
				if ($is_sublist || isset($expandlist[$child_tn_info['postid']])){
					$expand = true;
				} else {
					$expand = false;
				}
				$this->child_tns[$count] = new treenode ($child_tn_info, $is_sublist,
														 $expand, $expandlist, $depth+1);				
			}			
		}
	}

	function display_header($row, $is_sublist = false){		
	#is_sublist is false by default for displaying root node on main page at the beginning.
	
		/* steps
		 * 1. change bg color using given row
		 * 2. indent header based on its depth
		 * 3. display + or - or no button
		 * 4. display header
		 * 5. change row
		 * 6. recursive display_header for each child header
		 * 
		 * Note that header is not displayed if its a dummy (depth = -1).
		 */
		
		#alternate bgcolor.
		if ($this->depth > -1){
			echo "<tr><td bgcolor = ";
			if ($row % 2){
				echo "#cccccc>";
			} else {
				echo "#ffffff>";
			}
			
			#indent reply headers based on their depth.
			for ($i = 0; $i < $this->depth; $i++){
				echo "<img src = '/images/spacer.gif' height = '18' width = '18' alt = '' valign = 'bottom'/>";
			}
			
			#display + - or spacer
			if (!$is_sublist && $this->children && sizeof($this->child_tns)){
				#Nodes that have children are expanded. so display collapse button.
				echo "<a href = 'index.php?collapse=".$this->postid."#".$this->postid."'/><img src=
					   '/images/minus.gif' height = '18' width = '18' alt = 'Collapse Thread'
					   border = '0' /></a>\n";
			} else if (!$is_sublist && $this->children){
				#if on main pg, has children, but not expanded. so display expand button.
				echo "<a href = 'index.php?expand=".$this->postid."#".$this->postid."'/><img src=
		 			   '/images/plus.gif' height = '18' width = '18' alt = 'Expand Thread'
            	  	   border = '0' /></a>\n";
			} else {
				#if nodes that are either sublists or has no child. So no button.
				echo "<img src = '/images/spacer.gif' height = '18' width = '18' alt = '' valign = 'bottom'/>";
			}
			
			#display the actual header.
			echo "<a name = '".$this->postid."'><a href = '/forum/view_post.php?postid=
       		 	 ".$this->postid."'/>".$this->title." - ".$this->poster." -
			     ".reformat_date($this->dateposted)."</a></td></tr>";
   			#to switch row color
			$row++;
		}
		#now recursively display this node's child nodes.
		for ($i = 0; $i < sizeof($this->child_tns); $i++){
			$row = $this->child_tns[$i]->display_header($row, $is_sublist);
		}
		return $row;
	}
}

function display_tree($expandlist, $postid = 0, $row = 0){
	global $table_width;
	
	echo "<table width = ".$table_width." cellpadding=4 cellspacing=4>";
	
	/* Purpose of this function is to display the entire tree of all posted headers except itself.
	 * Since recursion takes only one parent node, parent node must have all main posts as its children
	 * in order to display the whole tree starting from and including all main posts.
	 * Although main posts do have real parent post, their parents are set to 0 by default.
	 * By creating a dummy treenode instance with its postid being 0, entire tree can be displayed.
	 * Upon calling display function of dummy instance, display_header starts from child treenodes 
	 * of dummy treenode which are main posts. Note that dummy treenode itself is not being displayed
	 * because its depth is -1.
	 */
	
	if ($postid > 0){
		$is_sublist = true;
	} else {
		$is_sublist = false;
	}
	
	#create a dummy treenode instance.
	$tn_info['postid'] = $postid;
	$tn_info['poster'] = '';
	$tn_info['title'] = '';
	$tn_info['dateposted'] = '0000-00-00 00:00:00';
	$tn_info['children'] = 1;
	$expand = true;
	$depth = -1;
	
	$dummy_tn = new treenode($tn_info, $is_sublist, $expand, $expandlist, $depth);
	
	$dummy_tn->display_header($row, $is_sublist);
	
	echo "</table>";
}

?>